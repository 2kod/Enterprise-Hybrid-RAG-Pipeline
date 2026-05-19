from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    top_k: int = Field(5, description="Number of documents to retrieve after reranking")

class Citation(BaseModel):
    source_id: str
    text_snippet: str
    metadata: dict = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    
class IngestResponse(BaseModel):
    message: str
    num_documents: int
