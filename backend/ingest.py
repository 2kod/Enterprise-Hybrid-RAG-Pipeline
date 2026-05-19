import os
import pickle
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_db"
BM25_PATH = "bm25_retriever.pkl"

# Direct LangChain wrapper around our local BGE Hugging Face model
class LocalBGEEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False).tolist()

def ingest_documents():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found in data directory. Make sure you have markdown files inside 'data/'!")
        return 0

    print(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Assign unique IDs to chunks for reference
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    print("Creating Vector Store with local BGE embeddings...")
    embeddings = LocalBGEEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()

    print("Creating BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"✅ Successfully ingested {len(chunks)} chunks locally.")
    return len(chunks)

if __name__ == "__main__":
    ingest_documents()