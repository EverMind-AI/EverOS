import asyncio
import os
import pytest
import numpy as np
from agentic_layer.vectorize_service import get_vectorize_service
from agentic_layer.rerank_service import get_rerank_service

# ===== Environment configuration =====
os.environ["VECTORIZE_PROVIDER"] = "vllm"
os.environ["VECTORIZE_BASE_URL"] = "http://localhost:11000/v1"
os.environ["VECTORIZE_MODEL"] = "Qwen3-Embedding-4B"
os.environ["VECTORIZE_DIMENSIONS"] = "1024"
os.environ["VECTORIZE_API_KEY"] = "EMPTY"

os.environ["RERANK_PROVIDER"] = "vllm"
os.environ["RERANK_BASE_URL"] = "http://localhost:12000/score"
os.environ["RERANK_MODEL"] = "Qwen3-Reranker-4B"
os.environ["RERANK_API_KEY"] = "EMPTY"

# os.environ["VECTORIZE_PROVIDER"] = "deepinfra"
# os.environ["VECTORIZE_BASE_URL"] = "https://api.deepinfra.com/v1/openai"
# os.environ["VECTORIZE_MODEL"] = "Qwen/Qwen3-Embedding-4B"
# os.environ["VECTORIZE_DIMENSIONS"] = "1024"

# os.environ["RERANK_PROVIDER"] = "deepinfra"
# os.environ["RERANK_BASE_URL"] = "https://api.deepinfra.com/v1/inference"
# os.environ["RERANK_MODEL"] = "Qwen/Qwen3-Reranker-4B"


@pytest.mark.asyncio
async def test_embedding():
    """Test Embedding and calculate similarity"""
    print("\n=== Test Embedding ===")
    
    # Define instruction (for query)
    query_task = "Given a search query, retrieve relevant passages that answer the query"
    
    # Prepare Query (user search query)
    query = "水果"
    
    # Prepare Documents (document content)
    doc1 = "苹果很好吃"
    doc2 = "香蕉也是水果" 
    doc3 = "汽车速度很快"
    
    print(f"Query Task: {query_task}")
    print(f"Query: {query}")
    print(f"Documents: [{doc1}, {doc2}, {doc3}]")

    service = get_vectorize_service()

    # Query: Use is_query=True
    print("\n--- Query Embedding (is_query=True) ---")
    query_emb = await service.get_embedding(query, instruction=query_task, is_query=True)
    print(f"Query vector dimension: {len(query_emb)}")
    print(f"Configured dimension: 1024")
    if len(query_emb) == 1024:
        print("✅ Query dimension correct")
    else:
        print(f"❌ Query dimension mismatch! Expected 1024, got {len(query_emb)}")

    # Documents: Use is_query=False (without instruction)
    print("\n--- Document Embeddings (is_query=False) ---")
    doc1_emb = await service.get_embedding(doc1, is_query=False)
    doc2_emb = await service.get_embedding(doc2, is_query=False)
    doc3_emb = await service.get_embedding(doc3, is_query=False)
    print(f"Document vector dimension: {len(doc1_emb)}")
    if len(doc1_emb) == 1024:
        print("✅ Document dimension correct")
    else:
        print(f"❌ Document dimension mismatch! Expected 1024, got {len(doc1_emb)}")
    
    # Verify all vector dimensions are consistent
    if len(query_emb) == len(doc1_emb) == len(doc2_emb) == len(doc3_emb) == 1024:
        print("\n✅ All vector dimensions match (1024)")
    else:
        print(f"\n❌ Vector dimensions inconsistent! Query:{len(query_emb)}, Doc1:{len(doc1_emb)}, Doc2:{len(doc2_emb)}, Doc3:{len(doc3_emb)}")
        return
    
    # Calculate similarity (Query vs Documents)
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim_q_doc1 = cos_sim(query_emb, doc1_emb)
    sim_q_doc2 = cos_sim(query_emb, doc2_emb)
    sim_q_doc3 = cos_sim(query_emb, doc3_emb)
    
    print(f"\nSimilarity results:")
    print(f"Query '{query}' vs Doc '{doc1}': {sim_q_doc1:.4f}")
    print(f"Query '{query}' vs Doc '{doc2}': {sim_q_doc2:.4f}")
    print(f"Query '{query}' vs Doc '{doc3}': {sim_q_doc3:.4f}")
    
    # Verify: doc2 ("香蕉也是水果") should be most relevant to query ("水果")
    if sim_q_doc2 > sim_q_doc1 and sim_q_doc2 > sim_q_doc3:
        print("✅ Similarity is normal ('香蕉也是水果' is most relevant to '水果')")
    else:
        print("⚠️  Similarity ranking does not fully match expectation")


@pytest.mark.asyncio
async def test_rerank():
    """Test Rerank"""
    print("\n=== Test Rerank ===")
    
    query = "苹果"
    instruction = "Given a question and a passage, determine if the passage contains information relevant to answering the question."
    
    docs = [
        {"episode": "苹果很好吃"},
        {"episode": "汽车很快"},
        {"episode": "香蕉也是水果"}
    ]
    
    print(f"Query: {query}")
    print(f"Instruction: {instruction}")
    
    # Call rerank
    service = get_rerank_service()
    async with service:
        results = await service.rerank_memories(query, docs, instruction)
    
    # Print results
    print("Rerank results:")
    for r in results:
        score = r.get('score', 0)
        text = r['episode']
        print(f"  {score:.4f} - {text}")


async def main():
    await test_embedding()
    await test_rerank()
    print("\n=== Test completed ===\n")


if __name__ == "__main__":
    asyncio.run(main())