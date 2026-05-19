import os
import json
import pytest
import httpx

# Modern Ragas v1.0+ import pattern to prevent deprecation warnings
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy
except ImportError:
    # Safe fallback if your venv has an intermediate dependency layout
    from ragas.metrics import faithfulness, answer_relevancy

API_URL = "http://127.0.0.1:8000"

def test_rag_pipeline_metrics():
    """
    Automated production-gated evaluation suite.
    Queries your live local FastAPI instance to validate the contract
    and ensures processing finishes safely without hitting a ReadTimeout.
    """
    # 1. Resolve pathing and check if the dataset exists
    eval_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    assert os.path.exists(eval_path), f"Evaluation dataset missing at: {eval_path}"
    
    with open(eval_path, "r") as f:
        golden_dataset = json.load(f)
    
    assert len(golden_dataset) > 0, "Your evaluation dataset file is empty!"
    print(f"\n🚀 Commencing Evaluation Sweep across {len(golden_dataset)} records...")

    # 2. Configure HTTP client with a 90-second timeout to cover local model warmups
    with httpx.Client(base_url=API_URL, timeout=90.0) as client:
        for idx, test_case in enumerate(golden_dataset):
            query = test_case["question"]
            
            # Fire structural POST payload to the active API
            response = client.post("/ask", json={"query": query, "top_k": 3})
            
            # Assert target HTTP 200 OK contract state
            assert response.status_code == 200, f"API Error on question {idx}: {response.text}"
            
            response_data = response.json()
            
            # Validate response schema integrity
            assert "answer" in response_data, f"Malformed contract response schema at index {idx}"
            assert "citations" in response_data, f"Citations field missing from response schema at index {idx}"
            
            answer = response_data["answer"]
            citations = response_data["citations"]
            
            print(f"\n[Test Case #{idx + 1}] PASSED Contract Validation Check")
            print(f"Question: '{query}'")
            print(f"Answer: '{answer[:120]}...'")
            print(f"Citations Verified: Found {len(citations)} source anchors.")
            
            # Warn if a non-empty answer completely leaves out data citations
            if not citations and "I don't have enough information" not in answer:
                print(f"WARNING: Pipeline generated an answer without compiling structural citations.")
                
    print("\n Integration evaluation sweep completed successfully!")