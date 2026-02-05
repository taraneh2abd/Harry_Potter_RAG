"""
Advanced Modular RAG Pipeline - Step 1: Document Processing and Indexing
تولید امبدینگ و ذخیره‌سازی برای جستجوی ترکیبی و بازرتبه‌بندی

Project: Advanced Modular RAG with Hybrid Search and Re-ranking
Based on: Comprehensive Survey of RAG (2024)
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------
# 1. توابع پردازش سند
# ---------------------------------------------------------------------

def load_pdf_text(pdf_path: str) -> str:
    """
    خواندن متن از فایل PDF
    
    Args:
        pdf_path: مسیر فایل PDF
        
    Returns:
        متن کامل PDF به صورت یک رشته
    """
    reader = PdfReader(pdf_path)
    pages_text = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    
    return "\n".join(pages_text)


def chunk_text_advanced(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    قطعه‌بندی پیشرفته متن با حفظ ساختار معنایی
    
    Args:
        text: متن کامل
        chunk_size: اندازه هر قطعه (تعداد کاراکتر)
        overlap: همپوشانی بین قطعات
        
    Returns:
        لیستی از قطعات متن
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # جداکننده‌های معنادار به ترتیب اولویت
    )
    
    return splitter.split_text(text)


def create_chroma_db(chunks, embeddings, db_path="./chroma_db"):
    """
    ایجاد پایگاه داده ChromaDB برای ذخیره برداری‌ها
    
    Args:
        chunks: لیست قطعات متن
        embeddings: آرایه numpy از امبدینگ‌ها
        db_path: مسیر ذخیره‌سازی
    """
    # ایجاد کلاینت ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # حذف collection قبلی اگر وجود دارد
    try:
        chroma_client.delete_collection("documents")
        print("✅ Collection قبلی حذف شد")
    except:
        pass
    
    # ایجاد collection جدید
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # ایجاد شناسه‌ها
    ids = [f"doc_{i}" for i in range(len(chunks))]
    
    # تبدیل embeddings به لیست (مورد نیاز ChromaDB)
    embeddings_list = embeddings.tolist()
    
    # افزودن اسناد به collection
    collection.add(
        embeddings=embeddings_list,
        documents=chunks,
        ids=ids,
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    
    print(f"✅ {len(chunks)} سند در ChromaDB ذخیره شد")
    print(f"   مسیر: {db_path}")


# ---------------------------------------------------------------------
# 2. کلاس‌های بازیابی و بازرتبه‌بندی (برای استفاده در مراحل بعد)
# ---------------------------------------------------------------------

class BM25Retriever:
    """پیاده‌سازی جستجوی کلیدواژه‌ای با BM25"""
    
    def __init__(self, documents):
        """
        Args:
            documents: لیست قطعات متن
        """
        import nltk
        from rank_bm25 import BM25Okapi
        
        self.documents = documents
        
        # توکنایز کردن اسناد
        self.tokenized_docs = [
            nltk.word_tokenize(doc.lower()) for doc in documents
        ]
        
        # ایجاد مدل BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def search(self, query, top_k=10):
        """
        جستجوی BM25
        
        Args:
            query: متن جستجو
            top_k: تعداد نتایج برتر
            
        Returns:
            لیستی از نتایج (متن و امتیاز)
        """
        import nltk
        
        # توکنایز query
        tokenized_query = nltk.word_tokenize(query.lower())
        
        # محاسبه امتیازها
        scores = self.bm25.get_scores(tokenized_query)
        
        # گرفتن top_k نتایج
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': scores[idx],
                'index': idx
            })
        
        return results


class HybridRetriever:
    """پیاده‌سازی جستجوی ترکیبی (Vector + BM25)"""
    
    def __init__(self, vector_index, chunks, model):
        """
        Args:
            vector_index: اندیس FAISS
            chunks: لیست قطعات متن
            model: مدل SentenceTransformer
        """
        self.vector_index = vector_index
        self.chunks = chunks
        self.model = model
        
        # ایجاد BM25 retriever
        self.bm25_retriever = BM25Retriever(chunks)
    
    def rrf_fusion(self, vector_indices, bm25_scores, k=10, constant=60):
        """
        ترکیب نتایج با Reciprocal Rank Fusion
        
        Args:
            vector_indices: اندیس‌های نتایج برداری
            bm25_scores: امتیازهای BM25 برای همه اسناد
            k: تعداد نتایج نهایی
            constant: ثابت RRF
            
        Returns:
            امتیازهای ترکیبی
        """
        # ایجاد دیکشنری برای ذخیره امتیازهای ترکیبی
        combined_scores = np.zeros(len(self.chunks))
        
        # افزودن امتیازهای برداری بر اساس رتبه
        for rank, idx in enumerate(vector_indices, start=1):
            combined_scores[idx] += 1 / (rank + constant)
        
        # پیدا کردن top_k نتایج BM25
        bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]
        
        # افزودن امتیازهای BM25 بر اساس رتبه
        for rank, idx in enumerate(bm25_top_indices, start=1):
            combined_scores[idx] += 1 / (rank + constant)
        
        return combined_scores
    
    def search(self, query, k=10, alpha=0.5):
        """
        جستجوی ترکیبی
        
        Args:
            query: متن جستجو
            k: تعداد نتایج
            alpha: وزن جستجوی برداری (0-1)
            
        Returns:
            لیست اسناد مرتبط
        """
        # ۱. جستجوی برداری
        query_embedding = self.model.encode([query])[0].astype("float32")
        D, I = self.vector_index.search(query_embedding.reshape(1, -1), k*2)
        
        # ۲. جستجوی BM25
        bm25_results = self.bm25_retriever.search(query, k*2)
        bm25_scores = np.zeros(len(self.chunks))
        for res in bm25_results:
            bm25_scores[res['index']] = res['score']
        
        # ۳. ترکیب با RRF
        combined_scores = self.rrf_fusion(I[0], bm25_scores, k)
        
        # ۴. بازگشت اسناد برتر
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        return [self.chunks[idx] for idx in top_indices]


class Reranker:
    """بازرتبه‌بندی نتایج با مدل Cross-Encoder"""
    
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        """
        Args:
            model_name: نام مدل بازرتبه‌بندی
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def rerank(self, query, documents, top_k=5):
        """
        بازرتبه‌بندی اسناد
        
        Args:
            query: متن جستجو
            documents: لیست اسناد
            top_k: تعداد نتایج نهایی
            
        Returns:
            لیست اسناد بازرتبه‌بندی شده
        """
        scores = []
        
        with torch.no_grad():
            for doc in documents:
                # encode کردن جفت query-document
                inputs = self.tokenizer.encode_plus(
                    query, doc,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                )
                
                # محاسبه امتیاز ارتباط
                score = self.model(**inputs).logits.squeeze().item()
                scores.append((score, doc))
        
        # مرتب‌سازی نزولی بر اساس امتیاز
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # بازگشت top_k اسناد
        return [doc for _, doc in scores[:top_k]]


# ---------------------------------------------------------------------
# 3. تابع اصلی پردازش و ذخیره‌سازی
# ---------------------------------------------------------------------

def main():
    """
    تابع اصلی: پردازش PDF، ایجاد امبدینگ و ذخیره‌سازی
    """
    print("=" * 60)
    print("پردازش سند و ایجاد پایگاه داده برای Advanced RAG")
    print("=" * 60)
    
    # ۱. خواندن PDF
    print("\n📖 مرحله ۱: خواندن فایل PDF...")
    text = load_pdf_text("Harry Potter - Book 1 - The Sorcerers Stone.pdf")
    print(f"   متن خوانده شده: {len(text):,} کاراکتر")
    
    # ۲. قطعه‌بندی پیشرفته
    print("\n✂️  مرحله ۲: قطعه‌بندی متن با Recursive Character Splitter...")
    chunks = chunk_text_advanced(text, chunk_size=800, overlap=150)
    print(f"   تعداد قطعات ایجاد شده: {len(chunks)}")
    
    # نمایش نمونه قطعات
    print("\n   نمونه قطعات:")
    for i in range(min(3, len(chunks))):
        print(f"   قطعه {i+1}: {chunks[i][:80]}...")
    
    # ۳. ایجاد امبدینگ‌ها
    print("\n🔢 مرحله ۳: ایجاد امبدینگ با مدل all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    batch_size = 32
    all_embeddings = []
    
    # ایجاد امبدینگ به صورت batch
    for i in tqdm(
        range(0, len(chunks), batch_size),
        desc="ایجاد امبدینگ",
        unit="batch"
    ):
        batch = chunks[i: i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)
    
    # ترکیب همه batchها
    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"   شکل embeddings: {embeddings.shape}")
    print(f"   بعد هر بردار: {embeddings.shape[1]}")
    
    # ۴. ذخیره در FAISS (برای جستجوی سریع)
    print("\n💾 مرحله ۴: ذخیره‌سازی در FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # استفاده از فاصله اقلیدسی
    index.add(embeddings)
    faiss.write_index(index, "hp_index.faiss")
    print(f"   تعداد بردارهای ذخیره شده: {index.ntotal}")
    
    # ۵. ذخیره در ChromaDB (مطابق پروژه)
    print("\n🗄️  مرحله ۵: ایجاد پایگاه داده ChromaDB...")
    create_chroma_db(chunks, embeddings)
    
    # ۶. ذخیره chunks (برای مراحل بعد)
    print("\n💾 مرحله ۶: ذخیره قطعات متن...")
    with open("hp_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("   فایل hp_chunks.pkl ذخیره شد")
    
    # ۷. ذخیره model و metadata
    print("\n📝 مرحله ۷: ذخیره metadata...")
    metadata = {
        'num_chunks': len(chunks),
        'embedding_dim': dim,
        'chunk_size': 800,
        'overlap': 150,
        'model_name': 'all-MiniLM-L6-v2'
    }
    
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("   فایل metadata.pkl ذخیره شد")
    
    # ۸. خلاصه نتایج
    print("\n" + "=" * 60)
    print("✅ پردازش با موفقیت کامل شد!")
    print("=" * 60)
    print("\n📊 خلاصه نتایج:")
    print(f"   • تعداد قطعات: {len(chunks)}")
    print(f"   • بعد embeddings: {dim}")
    print(f"   • حجم embeddings: {embeddings.shape}")
    print("\n📁 فایل‌های ایجاد شده:")
    print("   1. hp_index.faiss    (اندیس FAISS)")
    print("   2. hp_chunks.pkl     (قطعات متن)")
    print("   3. metadata.pkl      (اطلاعات پروژه)")
    print("   4. ./chroma_db/      (پایگاه داده ChromaDB)")
    print("\n🚀 آماده برای مرحله بعد: پیاده‌سازی Hybrid Search و Re-ranking")


# ---------------------------------------------------------------------
# اجرای برنامه
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # دانلود punkt برای nltk (فقط یک بار)
    import nltk
    nltk.download('punkt', quiet=True)
    
    # ایمپورت‌های مرتبط با transformers (در صورت نیاز)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        print("⚠️  کتابخانه transformers نصب نیست. برای re-ranking نیاز است.")
        print("   دستور نصب: pip install transformers torch")
    
    main()