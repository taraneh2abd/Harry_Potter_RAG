import faiss
# provie a DS similarity search approach
import re
from typing import List
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


# sentence_aware_chunking
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            overlap_sentences = []
            overlap_length = 0
            
            for sent in reversed(current_chunk):
                if overlap_length + len(sent) <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += len(sent) + 1
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_length = overlap_length + sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
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



