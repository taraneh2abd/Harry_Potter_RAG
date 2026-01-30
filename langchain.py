import pickle
import faiss

from langchain_community.vectorstores import FAISS
# FAISS: Stores and searches vector embeddings efficiently for similarity search (retrieval)

from langchain_community.embeddings import HuggingFaceEmbeddings
# HuggingFaceEmbeddings: Converts text into numerical vectors using a transformer model

from langchain_openai import ChatOpenAI
# ChatOpenAI: Connects to an OpenAI-compatible chat model (e.g. GPT-3.5 / GPT-4)

from langchain.schema import Document
# Document: Standard container for text data passed through LangChain

from langchain.chains import RetrievalQA
# RetrievalQA: Combines document retrieval with an LLM to answer questions using retrieved context

# ===================== CONFIG =====================

FAISS_INDEX_PATH = "hp_index.faiss"
CHUNKS_PATH = "hp_chunks.pkl"

OPENAI_API_KEY = "sk-REPLACE_ME"
OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"

TOP_K = 6

# ===================== LOAD EMBEDDINGS =====================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ===================== LOAD FAISS INDEX =====================

index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=dict(enumerate(documents)),
    index_to_docstore_id=dict(enumerate(range(len(documents))))
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)

# ===================== LOAD LLM =====================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# ===================== RAG CHAIN =====================

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ===================== RUN =====================

if __name__ == "__main__":

    question = "Why did Dumbledore leave Harry with the Dursleys instead of a wizarding family?"

    result = qa_chain.invoke(question)

    print("\n========== Retrieved Chunks ==========\n")

    for i, doc in enumerate(result["source_documents"], start=1):
        print("=" * 90)
        print(f"Chunk #{i}")
        print("=" * 90)
        print(doc.page_content)
        print()

    print("\n========== Final Answer ==========\n")
    print(result["result"])
