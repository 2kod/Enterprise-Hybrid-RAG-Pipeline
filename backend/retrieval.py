import os
import pickle
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder, SentenceTransformer

CHROMA_PERSIST_DIR = "chroma_db"
BM25_PATH = "bm25_retriever.pkl"

# Must match ingest.py exactly
class LocalBGEEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False).tolist()

class HybridRerankRetriever:
    def __init__(self):
        print("Initializing HybridRerankRetriever components...")
        
        # 1. Load Vector Store (BGE-small + Chroma)
        self.embeddings = LocalBGEEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 2. Load BM25 Retriever from disk
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            self.bm25_retriever.k = 10
            print("Successfully loaded BM25 Retriever from disk.")
        else:
            self.bm25_retriever = None
            print(f"WARNING: {BM25_PATH} not found. Running vector-only retrieval.")

        print("Loading local Cross-Encoder ('ms-marco-MiniLM')...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Retriever system is online and ready.")

    def _manual_rrf(self, vector_docs, bm25_docs, k=60):
        rrf_scores = {}
        for rank, doc in enumerate(vector_docs):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (k + (rank + 1))
            
        for rank, doc in enumerate(bm25_docs):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (k + (rank + 1))
            
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def retrieve_and_rerank(self, query: str, top_k: int = 5):
        vector_docs = self.vectorstore.similarity_search(query, k=10)
        
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            initial_docs = self._manual_rrf(vector_docs, bm25_docs)
        else:
            initial_docs = vector_docs
        
        if not initial_docs:
            return []

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]