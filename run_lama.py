import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re

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

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results


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

    question = "How many years older is Mr.Flamel than Harry Potter?"

    retrieved_chunks = vs.search(question, top_k=1)

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
