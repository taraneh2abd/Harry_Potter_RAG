import faiss
# provie a DS similarity search approach
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pypdf import PdfReader
from tqdm import tqdm


def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def main():
    text = load_pdf_text("Harry Potter - Book 1 - The Sorcerers Stone.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    batch_size = 32  #chunk in each model input 
    all_embeddings = []

    for i in tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding chunks",
        unit="batch"
    ):
        batch = chunks[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    # convert to vstack for FAISS -> becuase FAISS needs float

    dim = embeddings.shape[1]
    # size of each embed in my model -> dim = 384
    index = faiss.IndexFlatL2(dim)
    # l2 = فاصله اقلیدسسی
    # if normal -> same as cosine similarity (my model normal it itself)
    # if not -> L is important
    index.add(embeddings)

    faiss.write_index(index, "hp_index.faiss")
    with open("hp_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        # خود متنا

    print("Index saved. Total vectors:", index.ntotal)


if __name__ == "__main__":
    main()
