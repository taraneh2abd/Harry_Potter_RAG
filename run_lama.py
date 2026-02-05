import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity 

from llama_cpp import Llama

MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
SEPARATOR = "=" * 90


class VectorStore:
    def __init__(self):
        self.index = faiss.read_index("hp_index.faiss")
        with open("hp_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        
        self.inverted_index = self._build_inverted_index()
        
    def _build_inverted_index(self):
        inverted_index = {}
        for idx, chunk in enumerate(self.chunks):
            words = re.findall(r'\w+', chunk.lower())
            for word in set(words): 
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(idx)
        return inverted_index
    
    def search(self, query, top_k=5):
        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
    
    def lexical_search(self, query, top_k=5, use_tfidf=True):
        if use_tfidf:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            scores = []
            
            for idx in top_indices:
                results.append(self.chunks[idx])
                scores.append(float(similarities[idx]))
            
            return results, scores
        else:
            query_words = re.findall(r'\w+', query.lower())
            doc_scores = {}
            
            for word in query_words:
                if word in self.inverted_index:
                    for doc_idx in self.inverted_index[word]:
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            results = []
            scores = []
            
            for doc_idx, score in sorted_docs[:top_k]:
                results.append(self.chunks[doc_idx])
                scores.append(score / len(query_words)) 
            
            return results, scores
    
    def hybrid_search(self, query, top_k=5, alpha=0.5, 
                     use_rerank=True, rerank_top_n=20):
        semantic_results, semantic_scores = self._semantic_search_with_scores(query, top_k=rerank_top_n)
        
        lexical_results, lexical_scores = self.lexical_search(query, top_k=rerank_top_n)
        
        combined_results = self._combine_results(
            semantic_results, semantic_scores,
            lexical_results, lexical_scores,
            alpha, top_k
        )
        
        if use_rerank and len(combined_results) > 0:
            combined_results = self._rerank_results(query, combined_results)
        
        return combined_results[:top_k]
    
    def _semantic_search_with_scores(self, query, top_k=20):
        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb).astype("float32")
        
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        scores = []
        
        for dist, idx in zip(distances[0], indices[0]):
            results.append(self.chunks[idx])
            similarity_score = 1.0 / (1.0 + dist)
            scores.append(similarity_score)
        
        return results, scores
    
    def _combine_results(self, semantic_results, semantic_scores,
                        lexical_results, lexical_scores,
                        alpha, top_k):
        from collections import defaultdict
        
        doc_scores = defaultdict(float)
        doc_content = {} 
        
        for content, score in zip(semantic_results, semantic_scores):
            doc_scores[content] += alpha * score
            doc_content[content] = content
        
        for content, score in zip(lexical_results, lexical_scores):
            doc_scores[content] += (1 - alpha) * score
            doc_content[content] = content
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        combined = []
        for content, score in sorted_docs[:top_k * 2]: 
            combined.append({
                'text': content,
                'score': score,
                'semantic_contribution': alpha * next((s for c, s in zip(semantic_results, semantic_scores) 
                                                      if c == content), 0),
                'lexical_contribution': (1 - alpha) * next((s for c, s in zip(lexical_results, lexical_scores) 
                                                          if c == content), 0)
            })
        
        return combined
    
    def _rerank_results(self, query, results):
        try:
            from sentence_transformers import CrossEncoder
            
            query_embedding = self.model.encode(query)
            
            for result in results:
                doc_embedding = self.model.encode(result['text'])
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    doc_embedding.reshape(1, -1)
                )[0][0]
                
                result['rerank_score'] = 0.7 * similarity + 0.3 * result['score']
            
            results.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
            
        except Exception as e:
            print(f"Re-ranking failed, using original scores: {e}")
        
        return results
    
    def hybrid_search_simple(self, query, top_k=5, alpha=0.5):
        semantic_results, semantic_scores = self._semantic_search_with_scores(query, top_k=top_k * 2)
        
        lexical_results, lexical_scores = self.lexical_search(query, top_k=top_k * 2, use_tfidf=True)
        
        return self._reciprocal_rank_fusion(
            semantic_results, lexical_results, top_k=top_k
        )
    
    def _reciprocal_rank_fusion(self, list1, list2, top_k=5, k=60):
        from collections import defaultdict
        
        scores = defaultdict(float)
        
        for rank, doc in enumerate(list1):
            scores[doc] += 1.0 / (k + rank + 1)
        
        for rank, doc in enumerate(list2):
            scores[doc] += 1.0 / (k + rank + 1)
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in sorted_docs[:top_k]]
    
    def search_with_filters(self, query, filter_func=None, top_k=5):
        results = self.hybrid_search_simple(query, top_k=top_k * 2)
        if filter_func:
            results = [doc for doc in results if filter_func(doc)]
        
        return results[:top_k]
    
    def explain_search(self, query, top_k=3):
        hybrid_results = self.hybrid_search(query, top_k=top_k, alpha=0.5)
        
        explanations = []
        for i, result in enumerate(hybrid_results):
            explanation = {
                'rank': i + 1,
                'text': result['text'][:200] + "...",  
                'total_score': result['score'],
                'semantic_score': result.get('semantic_contribution', 'N/A'),
                'lexical_score': result.get('lexical_contribution', 'N/A'),
                'keywords_found': self._extract_matching_keywords(query, result['text'])
            }
            explanations.append(explanation)
        
        return explanations
    
    def _extract_matching_keywords(self, query, document):
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        common_words = query_words.intersection(doc_words)
        return list(common_words)[:10]


llm_model = Llama(
    model_path=r"C:\Users\T.Abdellahi\Downloads\llama-2-7b-chat.Q4_0.gguf",
    verbose=False,
    log_disable=True,
    n_gpu_layers=-1
)


def ask_llm_local(context: str, question: str, max_tokens=200) -> str:
    prompt = f"""
You must answer ONLY using the context below.
If the answer is not in the context, say "I don't know".
Your answer should be as short as possible with absolutely no introduction.
Context:
{context}

Question:
{question}

Answer (in English):
""".strip()

    out = llm_model(prompt, max_tokens=max_tokens)
    return out["choices"][0]["text"]

# you can have pos-Retrieval :     Analyze this query and create a search plan before retrieve
#  Pre-Retrieval Reasoning is used....
def ask_llm_local_with_reasoning(context: str, question: str) -> str:
    prompt = f"""
Before answering, briefly explain your reasoning.
If a calculation is required, make sure to explicitly use the words: add, subtract, multiply, divide.
- Identify relevant information from the context.
- Describe the processing you will do on it.
- Then give the final short answer.
Context:
{context}

Question:
{question}

Reasoning + Answer (in English, concise):
""".strip()

    out = llm_model(prompt, max_tokens=400)
    return out["choices"][0]["text"]


def calculator_from_text(text: str):
    """
    Look for patterns like 'subtract X from Y' or 'add X and Y'
    and compute the result.
    """
    match_sub = re.search(r"subtract (\d+) from (\d+)", text)
    if match_sub:
        x = int(match_sub.group(1))
        y = int(match_sub.group(2))
        return y - x

    match_add = re.search(r"add (\d+) and (\d+)", text)
    if match_add:
        x = int(match_add.group(1))
        y = int(match_add.group(2))
        return x + y

    match_mul = re.search(r"multiply (\d+) and (\d+)", text)
    if match_mul:
        x = int(match_mul.group(1))
        y = int(match_mul.group(2))
        return x * y

    match_div = re.search(r"divide (\d+) by (\d+)", text)
    if match_div:
        x = int(match_div.group(1))
        y = int(match_div.group(2))
        if y != 0:
            return x / y

    return None


if __name__ == "__main__":
    vs = VectorStore()

    # question = "What is the name of Harry Potter’s aunt and uncle?"
    # question = "Why  Flamel’s Sorcerer’s Stone is moved out from Gringotts!?"
    # question = "Why did Dumbledore leave Harry with the Dursleys instead of a wizarding family?"
    # question = "What does Harry’s experience with the Mirror of Erised reveal about his deepest desire?"
    question = "How many years older is Mr.Flamel than Harry Potter?"
    # question = "who is best friend of harry potter and worst enemy of him?"

    
# here you can add query understanding(pre-reasoning) here
# you can have hybrid search here instead of faiss

    # retrieved_chunks = vs.search(question, top_k=1)  
    retrieved_chunks = vs.hybrid_search(question, top_k=3, alpha=0.6) 
    print("\n Retrieved Chunks (before LLM call):\n")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(MAGENTA)
        print(SEPARATOR)
        print(f"Retrieved Chunk #{i}")
        print(SEPARATOR)
        print(chunk)
        print(SEPARATOR)
        print(RESET)
        print("\n")

    context = "\n\n".join(retrieved_chunks)
    context = context[:400]

    answer_with_reasoning = ask_llm_local_with_reasoning(context, question)

    calc_result = calculator_from_text(answer_with_reasoning)

    if calc_result is not None:
        print("\n>>>>>>>>> Reasoning + Calculated Result (via module):\n")
        print(RED)
        print(SEPARATOR)
        print(f"{answer_with_reasoning}\n\nCalculated Result: {calc_result}")
        print(SEPARATOR)
        print(RESET)

        final_prompt = f"""
You must answer ONLY using the context below and the calculated result provided.
If the answer is not in the context, say "I don't know".
Use the calculated result exactly as given.
Context:
{context}

Question:
{question}

Calculated Result: {calc_result}

Answer (in English, concise):
""".strip()

        final_answer = llm_model(final_prompt, max_tokens=200)["choices"][0]["text"]

        print("\n>>>>>>>>> Final Answer with Accurate Calculation:\n")
        print(CYAN)
        print(SEPARATOR)
        print(final_answer)
        print(SEPARATOR)
        print(RESET)

    else:
        print("\n>>>>>>>>> Reasoning + Final Answer:\n")
        print(CYAN)
        print(SEPARATOR)
        print(answer_with_reasoning)
        print(SEPARATOR)
        print(RESET)





