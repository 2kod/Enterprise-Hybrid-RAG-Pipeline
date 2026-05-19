# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from backend.models import QueryRequest, QueryResponse, IngestResponse
# pyrefly: ignore [missing-import]
from backend.retrieval import HybridRerankRetriever
# pyrefly: ignore [missing-import]
from backend.generation import RAGGenerator
from backend.ingest import ingest_documents
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="Ask My Docs - Enterprise RAG API")

# Initialize components lazily to avoid loading heavy models on import
retriever = None
generator = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = HybridRerankRetriever()
    return retriever

def get_generator():
    global generator
    if generator is None:
        generator = RAGGenerator()
    return generator


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        retriever_instance = get_retriever()
        generator_instance = get_generator()
        
        # 1. Retrieve & Rerank
        docs = retriever_instance.retrieve_and_rerank(request.query, top_k=request.top_k)
        
        # 2. Generate with Citations
        answer, citations = generator_instance.generate_answer(request.query, docs)
        
        return QueryResponse(answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def run_ingestion():
    try:
        num_docs = ingest_documents()
        # Force re-initialization of retriever to pick up new index
        global retriever
        retriever = None 
        return IngestResponse(
            message="Ingestion completed successfully.",
            num_documents=num_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
