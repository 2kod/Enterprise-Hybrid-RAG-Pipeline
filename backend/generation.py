import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

# Ensure the .env file is loaded and completely overrides empty shell sessions
load_dotenv(override=True)

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from backend.models import Citation

# Citation enforcement prompt template
PROMPT_TEMPLATE = """You are an expert AI assistant answering questions based strictly on the provided documents.
You must cite your sources for EVERY claim you make using the exact document IDs provided in the context.

Context documents are provided in the following format:
Document [ID]:
<Content>

Format your citations like this: [Doc ID]
If multiple documents support a claim, format it like this: [Doc ID1][Doc ID2]

If the answer is not contained in the provided documents, say "I don't have enough information to answer that based on the provided documents." DO NOT hallucinate.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_id = doc.metadata.get("chunk_id", f"unknown_{i}")
        formatted.append(f"Document [{doc_id}]:\n{doc.page_content}")
    return "\n\n".join(formatted)

class RAGGenerator:
    def __init__(self):
        # Fetch the api key safely from your local environment file string
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # FIX: The modern SDK expects the model string without 'models/' 
        # and we explicitly pass the api_key to the constructor parameters.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            api_key=api_key
        )
        
        self.chain = (
            prompt 
            | self.llm 
            | StrOutputParser()
        )

    def generate_answer(self, query: str, retrieved_docs: list) -> Tuple[str, List[Citation]]:
        if not retrieved_docs:
            return "I don't have enough information to answer that based on the provided documents.", []

        # Create mapping from doc_id to the actual document for citation generation
        doc_mapping = {
            doc.metadata.get("chunk_id", f"unknown_{i}"): doc
            for i, doc in enumerate(retrieved_docs)
        }

        # Run LLM Chain
        context_str = format_docs(retrieved_docs)
        response_text = self.chain.invoke({
            "context": context_str,
            "question": query
        })

        # Parse citations from output
        citation_pattern = r"\[(.*?)\]"
        cited_ids = set(re.findall(citation_pattern, response_text))
        
        citations = []
        for cited_id in cited_ids:
            if cited_id in doc_mapping:
                doc = doc_mapping[cited_id]
                citations.append(
                    Citation(
                        source_id=cited_id,
                        text_snippet=doc.page_content,
                        metadata=doc.metadata
                    )
                )

        return response_text, citations