import faiss
import numpy as np
import pickle
import re
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
SEPARATOR = "=" * 90


class HybridVectorStore:
    def __init__(self):
        self.index = faiss.read_index("hp_index.faiss")
        with open("hp_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.documents = [Document(page_content=chunk) for chunk in self.chunks]
        
        self.vector_store = FAISS.from_documents(
            self.documents, 
            self.embeddings
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=20
        )
        
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
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Semantic search using FAISS embeddings"""
        return self.vector_store.similarity_search(query, k=top_k)
    
    def semantic_search_with_scores(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search with similarity scores"""
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'text': doc.page_content,
                'score': 1.0 / (1.0 + score),
                'type': 'semantic'
            })
        return results
    
    def lexical_search(self, query: str, top_k: int = 5, use_tfidf: bool = True) -> List[Dict]:
        """Lexical search using TF-IDF or BM25"""
        if use_tfidf:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            
            for idx in top_indices:
                results.append({
                    'text': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'type': 'tfidf'
                })
            
            return results
        else:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
            results = []
            for i, doc in enumerate(bm25_docs):
                results.append({
                    'text': doc.page_content,
                    'score': 1.0 / (i + 1),
                    'type': 'bm25'
                })
            return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Hybrid search combining semantic and lexical results"""
        semantic_results = self.semantic_search_with_scores(query, top_k=top_k*2)
        lexical_results = self.lexical_search(query, top_k=top_k*2, use_tfidf=True)
        
        return self._reciprocal_rank_fusion(semantic_results, lexical_results, top_k=top_k)
    
    def _reciprocal_rank_fusion(self, list1: List[Dict], list2: List[Dict], 
                               top_k: int = 5, k: int = 60) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        from collections import defaultdict
        
        scores = defaultdict(float)
        doc_contents = {}
        
        for rank, result in enumerate(list1):
            doc_text = result['text']
            scores[doc_text] += 1.0 / (k + rank + 1)
            doc_contents[doc_text] = result
        
        for rank, result in enumerate(list2):
            doc_text = result['text']
            scores[doc_text] += 1.0 / (k + rank + 1)
            if doc_text not in doc_contents:
                doc_contents[doc_text] = result
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        combined_results = []
        for doc_text, score in sorted_docs[:top_k]:
            result = doc_contents[doc_text].copy()
            result['combined_score'] = score
            combined_results.append(result)
        
        return combined_results
    
    def create_ensemble_retriever(self, weights: List[float] = None) -> EnsembleRetriever:
        """Create an ensemble retriever for more advanced hybrid search"""
        if weights is None:
            weights = [0.5, 0.5]
        
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = self.bm25_retriever
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
        
        return ensemble_retriever
    
    def search_with_explanation(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search with detailed explanation of results"""
        hybrid_results = self.hybrid_search(query, top_k=top_k)
        
        explanations = []
        for i, result in enumerate(hybrid_results):
            explanation = {
                'rank': i + 1,
                'text': result['text'][:200] + "...",
                'total_score': result.get('combined_score', result.get('score', 0)),
                'search_type': result.get('type', 'hybrid'),
                'keywords_found': self._extract_matching_keywords(query, result['text'])
            }
            explanations.append(explanation)
        
        return explanations
    
    def _extract_matching_keywords(self, query: str, document: str) -> List[str]:
        """Extract common keywords between query and document"""
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        common_words = query_words.intersection(doc_words)
        return list(common_words)[:10]


class CustomPromptTemplate:
    """Custom prompt templates for different scenarios"""
    
    @staticmethod
    def get_qa_prompt():
        """Prompt for standard QA"""
        template = """
You must answer ONLY using the context below.
If the answer is not in the context, say "I don't know".
Your answer should be as short as possible with absolutely no introduction.

Context:
{context}

Question:
{question}

Answer (in English):
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_reasoning_prompt():
        """Prompt for reasoning-based QA"""
        template = """
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
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_calculation_prompt(calculation_result: float):
        """Prompt for QA with calculated result"""
        template = f"""
You must answer ONLY using the context below and the calculated result provided.
If the answer is not in the context, say "I don't know".
Use the calculated result exactly as given.

Context:
{{context}}

Question:
{{question}}

Calculated Result: {calculation_result}

Answer (in English, concise):
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )


def calculator_from_text(text: str) -> Optional[float]:
    """
    Extract and compute mathematical operations from text
    """
    patterns = [
        (r"subtract (\d+) from (\d+)", lambda x, y: y - x),
        (r"add (\d+) and (\d+)", lambda x, y: x + y),
        (r"multiply (\d+) and (\d+)", lambda x, y: x * y),
        (r"divide (\d+) by (\d+)", lambda x, y: x / y if y != 0 else None),
        (r"(\d+) minus (\d+)", lambda x, y: x - y),
        (r"(\d+) plus (\d+)", lambda x, y: x + y),
    ]
    
    for pattern, operation in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                return operation(x, y)
            except:
                continue
    
    return None


def initialize_llm():
    """Initialize the LLaMA model with LangChain"""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=r"C:\Users\T.Abdellahi\Downloads\llama-2-7b-chat.Q4_0.gguf",
        temperature=0.1,
        max_tokens=200,
        top_p=1,
        callback_manager=callback_manager,
        verbose=False,
        n_gpu_layers=-1,
        n_batch=512,
    )
    
    return llm


def main():
    vs = HybridVectorStore()
    llm = initialize_llm()
    
    questions = [
        "What is the name of Harry Potter’s aunt and uncle?",
        "Why was Flamel’s Sorcerer’s Stone moved out from Gringotts?",
        "Why did Dumbledore leave Harry with the Dursleys instead of a wizarding family?",
        "What does Harry’s experience with the Mirror of Erised reveal about his deepest desire?",
        "How many years older is Mr. Flamel than Harry Potter?",
        "Who is best friend of Harry Potter and worst enemy of him?"
    ]
    
    for question in questions:
        print(f"\n{'='*100}")
        print(f"Question: {question}")
        print(f"{'='*100}")
        
        retrieved_chunks = vs.hybrid_search(question, top_k=3, alpha=0.6)
        
        print("\n Retrieved Chunks (before LLM call):\n")
        for i, chunk_info in enumerate(retrieved_chunks, start=1):
            print(MAGENTA)
            print(SEPARATOR)
            print(f"Retrieved Chunk #{i} - Type: {chunk_info.get('type', 'unknown')}")
            print(f"Score: {chunk_info.get('combined_score', chunk_info.get('score', 0)):.4f}")
            print(SEPARATOR)
            print(chunk_info['text'])
            print(SEPARATOR)
            print(RESET)
            print("\n")
        
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        context = context[:400]
        
        reasoning_prompt = CustomPromptTemplate.get_reasoning_prompt()
        reasoning_chain = reasoning_prompt | llm
        answer_with_reasoning = reasoning_chain.invoke({
            "context": context,
            "question": question
        })
        
        calc_result = calculator_from_text(answer_with_reasoning)
        
        if calc_result is not None:
            print("\n>>>>>>>>> Reasoning + Calculated Result:\n")
            print(RED)
            print(SEPARATOR)
            print(f"{answer_with_reasoning}\n\nCalculated Result: {calc_result}")
            print(SEPARATOR)
            print(RESET)
            
            calculation_prompt = CustomPromptTemplate.get_calculation_prompt(calc_result)
            final_chain = calculation_prompt | llm
            final_answer = final_chain.invoke({
                "context": context,
                "question": question
            })
            
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
        
        print("\n" + "="*100)
        print("Search Explanation:")
        print("="*100)
        explanations = vs.search_with_explanation(question, top_k=2)
        for exp in explanations:
            print(f"\nRank {exp['rank']}:")
            print(f"Score: {exp['total_score']:.4f}")
            print(f"Type: {exp['search_type']}")
            print(f"Keywords found: {', '.join(exp['keywords_found'])}")
            print(f"Text preview: {exp['text']}")


if __name__ == "__main__":
    main()
