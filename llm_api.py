
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pypdf import PdfReader
from tqdm import tqdm

class VectorStore:
    def __init__(self):
        self.index = faiss.read_index("hp_index.faiss")
        with open("hp_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results

MAGENTA = "\033[95m"
RESET = "\033[0m"
SEPARATOR = "=" * 90


URL = "https://api.chatanywhere.tech/v1/chat/completions"


def ask_llm(context: str, question: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You must answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer (in English):
""".strip()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(URL, headers=headers, json=data)

    if response.status_code != 200:
        raise RuntimeError(f"LLM request failed: {response.status_code}\n{response.text}")

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    vs = VectorStore()

    # question = "What is the name of Harry Potter’s aunt and uncle?"
    question = "Why did Dumbledore leave Harry with the Dursleys instead of a wizarding family?"
    # question = "What does Harry’s experience with the Mirror of Erised reveal about his deepest desire?"
    # question = "Why was the Sorcerer’s Stone hidden at Hogwarts instead of Gringotts?"
    # question = "what is the most task of transformers?"

    retrieved_chunks = vs.search(question, top_k=6)

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

    answer = ask_llm(context, question)

    print("\n>>>>>>>>> Final Answer (from LLM):\n")
    print(answer)
