"""
Integration Test for Vectorize and Rerank Services with Real Configuration

Tests the embedding and reranking services using actual environment configuration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path (for running from tests directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from agentic_layer.vectorize_service import get_vectorize_service
from agentic_layer.rerank_service import get_rerank_service
from api_specs.memory_models import MemoryType
from agentic_layer.rerank_interface import RerankMemResponse

# Load environment variables
load_dotenv(project_root / ".env")


async def test_vectorize_service():
    """Test vectorization service with real configuration"""
    print("\n" + "=" * 80)
    print("🔹 Testing Vectorize Service")
    print("=" * 80)
    
    vectorize_service = get_vectorize_service()
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Provider: {os.getenv('VECTORIZE_PROVIDER', 'N/A')}")
    print(f"   Base URL: {os.getenv('VECTORIZE_BASE_URL', 'N/A')}")
    print(f"   Model: {os.getenv('VECTORIZE_MODEL', 'N/A')}")
    print(f"   Fallback Provider: {os.getenv('VECTORIZE_FALLBACK_PROVIDER', 'N/A')}")
    print(f"   Fallback Base URL: {os.getenv('VECTORIZE_FALLBACK_BASE_URL', 'N/A')}")
    print(f"   Dimensions: {os.getenv('VECTORIZE_DIMENSIONS', 'N/A')}")
    
    # Test queries
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Deep learning uses neural networks for pattern recognition"
    ]
    
    print(f"\n🧪 Testing with {len(test_texts)} texts...")
    
    try:
        # Test single embedding
        print("\n1️⃣ Testing single embedding...")
        single_embedding = await vectorize_service.get_embedding(test_texts[0])
        print(f"   ✅ Single embedding shape: {single_embedding.shape}")
        print(f"   ✅ First 5 values: {single_embedding[:5]}")
        print(f"   ✅ Norm: {(single_embedding ** 2).sum() ** 0.5:.4f}")
        
        # Test batch embeddings
        print("\n2️⃣ Testing batch embeddings...")
        batch_embeddings = await vectorize_service.get_embeddings(test_texts)
        print(f"   ✅ Batch embeddings count: {len(batch_embeddings)}")
        for i, emb in enumerate(batch_embeddings):
            print(f"   ✅ Text {i+1} shape: {emb.shape}, norm: {(emb ** 2).sum() ** 0.5:.4f}")
        
        # Test query embedding
        print("\n3️⃣ Testing query embedding...")
        query_embedding = await vectorize_service.get_embedding(
            "What is machine learning?", 
            is_query=True
        )
        print(f"   ✅ Query embedding shape: {query_embedding.shape}")
        print(f"   ✅ Query norm: {(query_embedding ** 2).sum() ** 0.5:.4f}")
        
        # Calculate similarities
        print("\n4️⃣ Testing similarity calculation...")
        similarities = []
        for i, doc_emb in enumerate(batch_embeddings):
            # Cosine similarity
            similarity = (query_embedding * doc_emb).sum() / (
                ((query_embedding ** 2).sum() ** 0.5) * ((doc_emb ** 2).sum() ** 0.5)
            )
            similarities.append((i, similarity))
            print(f"   📊 Text {i+1} similarity: {similarity:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"\n   🏆 Most relevant: Text {similarities[0][0]+1} (score: {similarities[0][1]:.4f})")
        
        print("\n✅ Vectorize service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Vectorize service test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vectorize_service.close()


async def test_rerank_service():
    """Test reranking service with real configuration"""
    print("\n" + "=" * 80)
    print("🔹 Testing Rerank Service")
    print("=" * 80)
    
    rerank_service = get_rerank_service()
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Provider: {os.getenv('RERANK_PROVIDER', 'N/A')}")
    print(f"   Base URL: {os.getenv('RERANK_BASE_URL', 'N/A')}")
    print(f"   Model: {os.getenv('RERANK_MODEL', 'N/A')}")
    print(f"   Fallback Provider: {os.getenv('RERANK_FALLBACK_PROVIDER', 'N/A')}")
    print(f"   Fallback Base URL: {os.getenv('RERANK_FALLBACK_BASE_URL', 'N/A')}")
    
    # Test query and documents
    query = "What is machine learning and how does it work?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data analysis, and automation.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It's particularly effective for image recognition and natural language processing.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It's used across industries for decision-making.",
        "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes that process information in layers."
    ]
    
    print(f"\n🧪 Testing with query and {len(documents)} documents...")
    print(f"   Query: '{query}'")
    
    # Prepare hits for rerank_memories method
    hits = []
    for idx, doc in enumerate(documents):
        hit = {
            "id": f"doc_{idx}",
            "_source": {"episode": doc},
            "memory_type": "episodic_memory",
            "score": 1.0
        }
        hits.append(hit)
    
    try:
        # Test memory reranking using rerank_memories
        print("\n1️⃣ Testing memory reranking (using rerank_memories)...")
        reranked_hits = await rerank_service.rerank_memories(query, hits)
        
        print(f"   ✅ Reranked {len(reranked_hits)} memories")
        print(f"\n   📊 Reranking results (sorted by relevance):")
        for i, hit in enumerate(reranked_hits[:5]):  # Show top 5
            doc_text = hit.get('_source', {}).get('episode', '')
            doc_preview = doc_text[:80] + "..." if len(doc_text) > 80 else doc_text
            score = hit.get('score', 0.0)
            print(f"   {i+1}. Score: {score:.4f}")
            print(f"      Text: {doc_preview}")
            print()
        
        # Verify ranking
        print("2️⃣ Verifying ranking quality...")
        top_hit = reranked_hits[0]
        top_doc = top_hit.get('_source', {}).get('episode', '')
        if "machine learning" in top_doc.lower():
            print("   ✅ Top result contains 'machine learning' - ranking is good!")
        else:
            print(f"   ⚠️  Top result doesn't mention machine learning explicitly")
            print(f"      (This might still be correct if using semantic similarity)")
        
        # Test with top_k
        print("\n3️⃣ Testing with top_k=3...")
        top_3_hits = await rerank_service.rerank_memories(query, hits, top_k=3)
        print(f"   ✅ Retrieved top {len(top_3_hits)} memories")
        for i, hit in enumerate(top_3_hits):
            score = hit.get('score', 0.0)
            print(f"   {i+1}. Score: {score:.4f}")
        
        print("\n✅ Rerank service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Rerank service test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await rerank_service.close()


async def test_retrieval_pipeline():
    """Test complete retrieval pipeline: embed + rerank"""
    print("\n" + "=" * 80)
    print("🔹 Testing Complete Retrieval Pipeline")
    print("=" * 80)
    
    vectorize_service = get_vectorize_service()
    rerank_service = get_rerank_service()
    
    # Test data
    query = "How does deep learning work?"
    documents = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Python programming language is known for its simplicity and readability, making it popular for beginners and experts alike.",
        "Machine learning algorithms can be supervised, unsupervised, or semi-supervised depending on the type of training data available.",
        "Neural networks consist of layers of interconnected nodes that process and transform information, mimicking the structure of the human brain.",
        "Data preprocessing is an essential step in machine learning that involves cleaning, transforming, and organizing data for analysis.",
    ]
    
    print(f"\n🧪 Testing retrieval pipeline...")
    print(f"   Query: '{query}'")
    print(f"   Documents: {len(documents)}")
    
    try:
        # Step 1: Generate embeddings
        print("\n📍 Step 1: Generate embeddings...")
        query_emb = await vectorize_service.get_embedding(query, is_query=True)
        doc_embs = await vectorize_service.get_embeddings(documents)
        print(f"   ✅ Query embedding: shape={query_emb.shape}")
        print(f"   ✅ Document embeddings: {len(doc_embs)} vectors")
        
        # Step 2: Calculate initial similarity scores
        print("\n📍 Step 2: Calculate similarity scores...")
        scores = []
        for i, doc_emb in enumerate(doc_embs):
            similarity = (query_emb * doc_emb).sum() / (
                ((query_emb ** 2).sum() ** 0.5) * ((doc_emb ** 2).sum() ** 0.5)
            )
            scores.append((i, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        print("   📊 Initial ranking (by embedding similarity):")
        for rank, (idx, score) in enumerate(scores[:3]):
            doc_preview = documents[idx][:60] + "..."
            print(f"   {rank+1}. Doc {idx+1}: {score:.4f} - {doc_preview}")
        
        # Step 3: Prepare hits and rerank using rerank_memories
        print("\n📍 Step 3: Rerank with reranker (using rerank_memories)...")
        
        # Prepare hits from documents with embedding scores
        hits = []
        for idx, score in scores:
            hit = {
                "id": f"doc_{idx}",
                "_source": {"episode": documents[idx]},
                "memory_type": "episodic_memory",
                "score": float(score)  # Use embedding similarity as initial score
            }
            hits.append(hit)
        
        # Rerank using rerank_memories
        reranked_hits = await rerank_service.rerank_memories(query, hits, top_k=3)
        
        print("   📊 Final ranking (after reranking):")
        for rank, hit in enumerate(reranked_hits):
            doc_text = hit.get('_source', {}).get('episode', '')
            doc_preview = doc_text[:60] + "..."
            score = hit.get('score', 0.0)
            print(f"   {rank+1}. Score: {score:.4f} - {doc_preview}")
        
        # Compare rankings
        print("\n📍 Step 4: Compare rankings...")
        initial_top_idx = scores[0][0]
        reranked_top_text = reranked_hits[0].get('_source', {}).get('episode', '')
        
        if documents[initial_top_idx] == reranked_top_text:
            print("   ✅ Rankings agree - top result is the same")
        else:
            print("   ℹ️  Rankings differ - reranker provided different ordering")
            print("      This is normal as reranker uses more sophisticated cross-attention")
        
        print("\n✅ Complete pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vectorize_service.close()
        await rerank_service.close()


async def test_compare_vllm_deepinfra_rerank():
    """Compare vLLM and DeepInfra rerank results for the same query"""
    print("\n" + "=" * 80)
    print("🔹 Comparing vLLM vs DeepInfra Rerank Results")
    print("=" * 80)
    
    # Import service classes directly
    from agentic_layer.rerank_vllm import VllmRerankService, VllmRerankConfig
    from agentic_layer.rerank_deepinfra import DeepInfraRerankService, DeepInfraRerankConfig
    
    # Test query and documents
    query = "What is machine learning and how does it work?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Neural networks are computational models inspired by the human brain."
    ]
    
    print(f"\n🧪 Testing with same query and {len(documents)} documents...")
    print(f"   Query: '{query}'")
    
    try:
        # Create vLLM service
        vllm_config = VllmRerankConfig(
            api_key=os.getenv("RERANK_API_KEY", "EMPTY"),
            base_url=os.getenv("RERANK_BASE_URL", "http://localhost:12000/v1/rerank"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        vllm_service = VllmRerankService(vllm_config)
        
        # Create DeepInfra service
        deepinfra_config = DeepInfraRerankConfig(
            api_key=os.getenv("RERANK_FALLBACK_API_KEY", ""),
            base_url=os.getenv("RERANK_FALLBACK_BASE_URL", "https://api.deepinfra.com/v1/inference"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        deepinfra_service = DeepInfraRerankService(deepinfra_config)
        
        print(f"\n📋 Service Configuration:")
        print(f"   vLLM:      {vllm_config.base_url}")
        print(f"   DeepInfra: {deepinfra_config.base_url}")
        print(f"   Model:     {vllm_config.model}")
        
        # Prepare hits for rerank_memories method
        # Format: List[Dict] with _source and memory_type
        hits = []
        for idx, doc in enumerate(documents):
            hit = {
                "id": f"test_doc_{idx}",
                "_source": {"episode": doc},
                "memory_type": "episodic_memory",
                "score": 1.0  # Initial score
            }
            hits.append(hit)
        
        # Test vLLM using rerank_memories
        print("\n1️⃣ Testing vLLM reranking (using rerank_memories)...")
        vllm_results = await vllm_service.rerank_memories(query, hits)
        print(f"   ✅ vLLM returned {len(vllm_results)} results")
        
        # Test DeepInfra using rerank_memories
        print("\n2️⃣ Testing DeepInfra reranking (using rerank_memories)...")
        deepinfra_results = await deepinfra_service.rerank_memories(query, hits)
        print(f"   ✅ DeepInfra returned {len(deepinfra_results)} results")
        
        # Compare results
        print("\n3️⃣ Comparing results...")
        print("\n   📊 vLLM Rankings:")
        for i, hit in enumerate(vllm_results[:5]):
            doc_text = hit.get('_source', {}).get('episode', '')[:60] + "..."
            score = hit.get('score', 0.0)
            print(f"   {i+1}. Score: {score:.4f} | {doc_text}")
        
        print("\n   📊 DeepInfra Rankings:")
        for i, hit in enumerate(deepinfra_results[:5]):
            doc_text = hit.get('_source', {}).get('episode', '')[:60] + "..."
            score = hit.get('score', 0.0)
            print(f"   {i+1}. Score: {score:.4f} | {doc_text}")
        
        # Calculate ranking correlation
        print("\n4️⃣ Analyzing ranking consistency...")
        
        # Get top document from each service
        vllm_top_doc = vllm_results[0].get('_source', {}).get('episode', '')
        deepinfra_top_doc = deepinfra_results[0].get('_source', {}).get('episode', '')
        
        if vllm_top_doc == deepinfra_top_doc:
            print("   ✅ Top ranked document is the SAME across both services")
        else:
            print("   ℹ️  Top ranked documents DIFFER between services")
            print(f"      This is expected as different implementations may have slight variations")
        
        # Calculate score correlation for top 3
        print("\n   📈 Score comparison (Top 3):")
        for i in range(min(3, len(vllm_results), len(deepinfra_results))):
            vllm_score = vllm_results[i].get('score', 0.0)
            deepinfra_score = deepinfra_results[i].get('score', 0.0)
            diff = abs(vllm_score - deepinfra_score)
            print(f"   Doc {i+1}: vLLM={vllm_score:.4f}, DeepInfra={deepinfra_score:.4f}, diff={diff:.4f}")
        
        # Check if rankings are similar (allowing for small variations)
        print("\n5️⃣ Ranking similarity analysis...")
        
        # Extract document texts from results
        vllm_docs = [hit.get('_source', {}).get('episode', '') for hit in vllm_results]
        deepinfra_docs = [hit.get('_source', {}).get('episode', '') for hit in deepinfra_results]
        
        # Find indices in original documents list
        vllm_indices = []
        for doc_text in vllm_docs:
            if doc_text in documents:
                vllm_indices.append(documents.index(doc_text))
        
        deepinfra_indices = []
        for doc_text in deepinfra_docs:
            if doc_text in documents:
                deepinfra_indices.append(documents.index(doc_text))
        
        # Compare top 3 rankings
        vllm_top3 = set(vllm_indices[:3])
        deepinfra_top3 = set(deepinfra_indices[:3])
        overlap = vllm_top3.intersection(deepinfra_top3)
        overlap_rate = len(overlap) / 3 if len(vllm_top3) >= 3 and len(deepinfra_top3) >= 3 else 0
        
        print(f"   Top-3 overlap: {len(overlap)}/3 documents ({overlap_rate*100:.0f}%)")
        
        if overlap_rate >= 0.67:  # At least 2 out of 3 match
            print("   ✅ Rankings are highly consistent between services")
        else:
            print("   ⚠️  Rankings show significant differences")
            print("      This may be due to model version differences or API variations")
        
        print("\n✅ Comparison test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vllm_service.close()
        await deepinfra_service.close()


async def test_compare_vllm_deepinfra_embedding():
    """Compare vLLM and DeepInfra embedding results for multiple queries"""
    print("\n" + "=" * 80)
    print("🔹 Comparing vLLM vs DeepInfra Embedding Results")
    print("=" * 80)
    
    # Import service classes directly
    from agentic_layer.vectorize_vllm import VllmVectorizeService, VllmVectorizeConfig
    from agentic_layer.vectorize_deepinfra import DeepInfraVectorizeService, DeepInfraVectorizeConfig
    import numpy as np
    
    # Test queries (at least 5 diverse queries)
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the concept of deep learning neural networks",
        "How to build a web application using Python",
        "What are the benefits of cloud computing?",
        "Describe the process of natural language processing",
        "How does blockchain technology ensure security?",
        "What is the difference between AI and machine learning?",
    ]
    
    print(f"\n🧪 Testing with {len(test_queries)} diverse queries...")
    
    try:
        # Create vLLM service
        vllm_config = VllmVectorizeConfig(
            api_key=os.getenv("VECTORIZE_API_KEY", "EMPTY"),
            base_url=os.getenv("VECTORIZE_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B"),
            dimensions=int(os.getenv("VECTORIZE_DIMENSIONS", "1024")),
        )
        vllm_service = VllmVectorizeService(vllm_config)
        
        # Create DeepInfra service
        deepinfra_config = DeepInfraVectorizeConfig(
            api_key=os.getenv("VECTORIZE_FALLBACK_API_KEY", ""),
            base_url=os.getenv("VECTORIZE_FALLBACK_BASE_URL", "https://api.deepinfra.com/v1/openai"),
            model=os.getenv("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B"),
            dimensions=int(os.getenv("VECTORIZE_DIMENSIONS", "1024")),
        )
        deepinfra_service = DeepInfraVectorizeService(deepinfra_config)
        
        print(f"\n📋 Service Configuration:")
        print(f"   vLLM:      {vllm_config.base_url}")
        print(f"   DeepInfra: {deepinfra_config.base_url}")
        print(f"   Model:     {vllm_config.model}")
        print(f"   Dimensions: {vllm_config.dimensions}")
        
        # Test each query
        print("\n1️⃣ Generating embeddings for all queries...")
        
        vllm_embeddings = []
        deepinfra_embeddings = []
        
        for i, query in enumerate(test_queries):
            # Get vLLM embedding
            vllm_emb = await vllm_service.get_embedding(query, is_query=True)
            vllm_embeddings.append(vllm_emb)
            
            # Get DeepInfra embedding
            deepinfra_emb = await deepinfra_service.get_embedding(query, is_query=True)
            deepinfra_embeddings.append(deepinfra_emb)
            
            print(f"   Query {i+1}: ✅ vLLM dim={vllm_emb.shape[0]}, DeepInfra dim={deepinfra_emb.shape[0]}")
        
        # Compare embeddings
        print("\n2️⃣ Comparing embedding properties...")
        
        # Check dimensions
        print("\n   📏 Dimension Check:")
        all_same_dim = all(
            vllm_emb.shape[0] == deepinfra_emb.shape[0] 
            for vllm_emb, deepinfra_emb in zip(vllm_embeddings, deepinfra_embeddings)
        )
        if all_same_dim:
            print(f"   ✅ All embeddings have the same dimension: {vllm_embeddings[0].shape[0]}")
        else:
            print(f"   ⚠️  Dimension mismatch detected!")
        
        # Compare norms
        print("\n   📊 Norm Comparison:")
        for i, (vllm_emb, deepinfra_emb) in enumerate(zip(vllm_embeddings, deepinfra_embeddings)):
            vllm_norm = np.linalg.norm(vllm_emb)
            deepinfra_norm = np.linalg.norm(deepinfra_emb)
            print(f"   Query {i+1}: vLLM={vllm_norm:.4f}, DeepInfra={deepinfra_norm:.4f}")
        
        # Calculate cosine similarity between same queries from different services
        print("\n3️⃣ Cross-service similarity analysis...")
        print("\n   🔄 Cosine Similarity (vLLM vs DeepInfra for same query):")
        
        similarities = []
        for i, (vllm_emb, deepinfra_emb) in enumerate(zip(vllm_embeddings, deepinfra_embeddings)):
            # Cosine similarity
            similarity = np.dot(vllm_emb, deepinfra_emb) / (
                np.linalg.norm(vllm_emb) * np.linalg.norm(deepinfra_emb)
            )
            similarities.append(similarity)
            query_preview = test_queries[i][:50] + "..." if len(test_queries[i]) > 50 else test_queries[i]
            print(f"   Query {i+1}: {similarity:.4f} | {query_preview}")
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        print(f"\n   📈 Statistics:")
        print(f"      Average similarity: {avg_similarity:.4f}")
        print(f"      Min similarity:     {min_similarity:.4f}")
        print(f"      Max similarity:     {max_similarity:.4f}")
        
        # Analyze element-wise differences
        print("\n4️⃣ Element-wise difference analysis...")
        
        total_diffs = []
        for i, (vllm_emb, deepinfra_emb) in enumerate(zip(vllm_embeddings, deepinfra_embeddings)):
            diff = np.abs(vllm_emb - deepinfra_emb)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            total_diffs.append(mean_diff)
            
            if i < 3:  # Show details for first 3 queries
                print(f"   Query {i+1}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}")
        
        avg_element_diff = np.mean(total_diffs)
        print(f"\n   Average element-wise difference: {avg_element_diff:.6f}")
        
        # Test embedding consistency within each service
        print("\n5️⃣ Consistency check (same query, multiple calls)...")
        
        test_query = test_queries[0]
        print(f"   Testing query: '{test_query[:60]}...'")
        
        # Get 3 embeddings from vLLM for the same query
        vllm_consistency = []
        for _ in range(3):
            emb = await vllm_service.get_embedding(test_query, is_query=True)
            vllm_consistency.append(emb)
        
        # Calculate pairwise similarities
        vllm_sim_1_2 = np.dot(vllm_consistency[0], vllm_consistency[1]) / (
            np.linalg.norm(vllm_consistency[0]) * np.linalg.norm(vllm_consistency[1])
        )
        vllm_sim_1_3 = np.dot(vllm_consistency[0], vllm_consistency[2]) / (
            np.linalg.norm(vllm_consistency[0]) * np.linalg.norm(vllm_consistency[2])
        )
        
        print(f"   vLLM consistency (call 1 vs 2): {vllm_sim_1_2:.6f}")
        print(f"   vLLM consistency (call 1 vs 3): {vllm_sim_1_3:.6f}")
        
        if vllm_sim_1_2 > 0.9999 and vllm_sim_1_3 > 0.9999:
            print(f"   ✅ vLLM produces highly consistent embeddings")
        else:
            print(f"   ⚠️  vLLM shows some variability")
        
        # Summary and conclusion
        print("\n6️⃣ Summary and Conclusion...")
        
        if avg_similarity > 0.99:
            print(f"   ✅ Embeddings are HIGHLY SIMILAR (avg={avg_similarity:.4f})")
            print(f"      Services are likely using the same model checkpoint")
        elif avg_similarity > 0.95:
            print(f"   ✅ Embeddings are VERY SIMILAR (avg={avg_similarity:.4f})")
            print(f"      Small differences likely due to implementation details")
        elif avg_similarity > 0.90:
            print(f"   ⚠️  Embeddings are MODERATELY SIMILAR (avg={avg_similarity:.4f})")
            print(f"      May indicate different model versions or post-processing")
        else:
            print(f"   ⚠️  Embeddings show SIGNIFICANT DIFFERENCES (avg={avg_similarity:.4f})")
            print(f"      Likely different model versions or configurations")
        
        print(f"\n   💡 Recommendation:")
        if avg_similarity > 0.95:
            print(f"      Safe to use either service for retrieval")
            print(f"      Fallback between services should work well")
        else:
            print(f"      Consider using a single service for consistency")
            print(f"      Fallback may produce different retrieval results")
        
        print("\n✅ Embedding comparison test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vllm_service.close()
        await deepinfra_service.close()


async def test_detailed_rerank_comparison():
    """Detailed comparison of vLLM vs DeepInfra rerank across multiple test cases"""
    print("\n" + "=" * 80)
    print("🔹 Detailed Rerank Comparison: Multiple Test Cases")
    print("=" * 80)
    
    # Import service classes
    from agentic_layer.rerank_vllm import VllmRerankService, VllmRerankConfig
    from agentic_layer.rerank_deepinfra import DeepInfraRerankService, DeepInfraRerankConfig
    import numpy as np
    
    # Define multiple test cases across different domains
    test_cases = [
        {
            "name": "Technology - Machine Learning",
            "query": "What is machine learning and how does it work?",
            "documents": [
                "Machine learning is a subset of AI that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns.",
                "Python is a programming language widely used in web development and data analysis.",
                "Deep learning is a subset of machine learning using neural networks with multiple layers.",
                "Cloud computing provides on-demand access to computing resources over the internet.",
                "Neural networks are computational models inspired by the human brain structure.",
            ]
        },
        {
            "name": "Healthcare - Diabetes",
            "query": "What are the symptoms and treatment of type 2 diabetes?",
            "documents": [
                "Type 2 diabetes symptoms include increased thirst, frequent urination, and unexplained weight loss. Treatment involves medication, diet, and exercise.",
                "Regular exercise helps maintain cardiovascular health and reduces stress levels.",
                "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin.",
                "A balanced diet rich in vegetables and whole grains promotes overall health.",
                "Diabetes complications can include heart disease, kidney damage, and vision problems if left untreated.",
            ]
        },
        {
            "name": "Finance - Stock Market",
            "query": "How does the stock market work and what factors affect stock prices?",
            "documents": [
                "Stock markets facilitate buying and selling of company shares. Prices are affected by supply, demand, company performance, and economic indicators.",
                "Cryptocurrency trading has grown popular with blockchain technology enabling decentralized transactions.",
                "Interest rates set by central banks influence borrowing costs and investment decisions.",
                "Real estate investment can provide steady income through rental properties.",
                "Stock market volatility increases during economic uncertainty, causing price fluctuations.",
            ]
        },
        {
            "name": "Climate - Global Warming",
            "query": "What causes global warming and what are its effects?",
            "documents": [
                "Global warming is caused by greenhouse gas emissions from burning fossil fuels. Effects include rising temperatures, melting ice caps, and extreme weather.",
                "Renewable energy sources like solar and wind power reduce carbon emissions.",
                "Ocean acidification occurs when seawater absorbs excess carbon dioxide from the atmosphere.",
                "Deforestation reduces the planet's capacity to absorb CO2, contributing to climate change.",
                "Electric vehicles produce zero direct emissions and help reduce air pollution.",
            ]
        },
        {
            "name": "Programming - Web Development",
            "query": "What are the best practices for building modern web applications?",
            "documents": [
                "Modern web development uses frameworks like React, Vue, or Angular. Best practices include responsive design, performance optimization, and security.",
                "Mobile app development requires knowledge of iOS and Android platforms.",
                "Database optimization improves query performance and reduces server load.",
                "Web applications should implement proper authentication, input validation, and secure data transmission.",
                "Version control systems like Git help teams collaborate on code efficiently.",
            ]
        },
        {
            "name": "History - World War II",
            "query": "What were the main causes and consequences of World War II?",
            "documents": [
                "World War II was caused by factors including the Treaty of Versailles, economic depression, and rise of totalitarian regimes. Consequences included millions of deaths and geopolitical changes.",
                "The Cold War was a period of geopolitical tension between the Soviet Union and the United States.",
                "The Industrial Revolution transformed economies from agriculture-based to manufacturing-focused.",
                "World War II ended with the atomic bombings of Hiroshima and Nagasaki, leading to Japan's surrender.",
                "The United Nations was established after WWII to promote international cooperation and prevent future conflicts.",
            ]
        },
        {
            "name": "Nutrition - Healthy Diet",
            "query": "What constitutes a healthy balanced diet?",
            "documents": [
                "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. It provides essential nutrients for optimal health.",
                "Regular physical activity combined with proper nutrition helps maintain healthy body weight.",
                "Processed foods high in sugar and saturated fats should be limited in a healthy diet.",
                "Hydration is important; adults should drink 6-8 glasses of water daily.",
                "Vitamins and minerals from diverse food sources support immune function and overall wellbeing.",
            ]
        },
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} different scenarios...")
    
    try:
        # Create services
        vllm_config = VllmRerankConfig(
            api_key=os.getenv("RERANK_API_KEY", "EMPTY"),
            base_url=os.getenv("RERANK_BASE_URL", "http://localhost:12000/v1/rerank"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        vllm_service = VllmRerankService(vllm_config)
        
        deepinfra_config = DeepInfraRerankConfig(
            api_key=os.getenv("RERANK_FALLBACK_API_KEY", ""),
            base_url=os.getenv("RERANK_FALLBACK_BASE_URL", "https://api.deepinfra.com/v1/inference"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        deepinfra_service = DeepInfraRerankService(deepinfra_config)
        
        print(f"\n📋 Services:")
        print(f"   vLLM:      {vllm_config.base_url}")
        print(f"   DeepInfra: {deepinfra_config.base_url}")
        
        # Store results for analysis
        all_results = []
        
        # Test each case
        for case_idx, test_case in enumerate(test_cases):
            print(f"\n{'='*80}")
            print(f"Test Case {case_idx + 1}: {test_case['name']}")
            print(f"{'='*80}")
            print(f"Query: {test_case['query']}")
            print(f"Documents: {len(test_case['documents'])}")
            
            # Prepare hits
            hits = []
            for doc_idx, doc in enumerate(test_case['documents']):
                hit = {
                    "id": f"case{case_idx}_doc{doc_idx}",
                    "_source": {"episode": doc},
                    "memory_type": "episodic_memory",
                    "score": 1.0
                }
                hits.append(hit)
            
            # Get rerank results from both services
            print(f"\n⏳ Reranking with vLLM...")
            vllm_results = await vllm_service.rerank_memories(test_case['query'], hits.copy())
            
            print(f"⏳ Reranking with DeepInfra...")
            deepinfra_results = await deepinfra_service.rerank_memories(test_case['query'], hits.copy())
            
            # Extract rankings and scores
            vllm_docs = [hit.get('_source', {}).get('episode', '') for hit in vllm_results]
            vllm_scores = [hit.get('score', 0.0) for hit in vllm_results]
            
            deepinfra_docs = [hit.get('_source', {}).get('episode', '') for hit in deepinfra_results]
            deepinfra_scores = [hit.get('score', 0.0) for hit in deepinfra_results]
            
            # Map back to original indices
            vllm_order = [test_case['documents'].index(doc) for doc in vllm_docs if doc in test_case['documents']]
            deepinfra_order = [test_case['documents'].index(doc) for doc in deepinfra_docs if doc in test_case['documents']]
            
            # Print results
            print(f"\n📊 vLLM Rankings:")
            for i, (doc, score) in enumerate(zip(vllm_docs, vllm_scores)):
                doc_preview = doc[:70] + "..." if len(doc) > 70 else doc
                orig_idx = test_case['documents'].index(doc) if doc in test_case['documents'] else -1
                print(f"   {i+1}. [Doc {orig_idx}] Score: {score:.4f} | {doc_preview}")
            
            print(f"\n📊 DeepInfra Rankings:")
            for i, (doc, score) in enumerate(zip(deepinfra_docs, deepinfra_scores)):
                doc_preview = doc[:70] + "..." if len(doc) > 70 else doc
                orig_idx = test_case['documents'].index(doc) if doc in test_case['documents'] else -1
                print(f"   {i+1}. [Doc {orig_idx}] Score: {score:.4f} | {doc_preview}")
            
            # Calculate metrics
            print(f"\n📈 Analysis:")
            
            # 1. Score comparison
            print(f"\n   Score Comparison:")
            score_diffs = []
            for i in range(min(len(vllm_scores), len(deepinfra_scores))):
                diff = abs(vllm_scores[i] - deepinfra_scores[i])
                score_diffs.append(diff)
                print(f"   Position {i+1}: vLLM={vllm_scores[i]:.4f}, DeepInfra={deepinfra_scores[i]:.4f}, diff={diff:.4f}")
            
            avg_score_diff = np.mean(score_diffs) if score_diffs else 0
            max_score_diff = np.max(score_diffs) if score_diffs else 0
            print(f"   Avg score difference: {avg_score_diff:.4f}")
            print(f"   Max score difference: {max_score_diff:.4f}")
            
            # 2. Ranking consistency
            print(f"\n   Ranking Consistency:")
            print(f"   vLLM order:      {vllm_order}")
            print(f"   DeepInfra order: {deepinfra_order}")
            
            # Top-1 match
            top1_match = vllm_order[0] == deepinfra_order[0] if vllm_order and deepinfra_order else False
            print(f"   Top-1 match: {'✅ YES' if top1_match else '❌ NO'}")
            
            # Top-3 overlap
            vllm_top3 = set(vllm_order[:3])
            deepinfra_top3 = set(deepinfra_order[:3])
            top3_overlap = len(vllm_top3.intersection(deepinfra_top3))
            print(f"   Top-3 overlap: {top3_overlap}/3 documents")
            
            # Kendall's Tau (rank correlation)
            from scipy.stats import kendalltau
            if len(vllm_order) == len(deepinfra_order):
                tau, p_value = kendalltau(vllm_order, deepinfra_order)
                print(f"   Kendall's Tau: {tau:.4f} (p={p_value:.4f})")
            
            # Spearman's correlation on scores
            from scipy.stats import spearmanr
            if len(vllm_scores) >= 2 and len(deepinfra_scores) >= 2:
                # Create score mapping for original documents
                vllm_score_map = {test_case['documents'].index(doc): score 
                                 for doc, score in zip(vllm_docs, vllm_scores) 
                                 if doc in test_case['documents']}
                deepinfra_score_map = {test_case['documents'].index(doc): score 
                                      for doc, score in zip(deepinfra_docs, deepinfra_scores) 
                                      if doc in test_case['documents']}
                
                # Get scores in original document order
                vllm_scores_ordered = [vllm_score_map.get(i, 0) for i in range(len(test_case['documents']))]
                deepinfra_scores_ordered = [deepinfra_score_map.get(i, 0) for i in range(len(test_case['documents']))]
                
                rho, p_value = spearmanr(vllm_scores_ordered, deepinfra_scores_ordered)
                print(f"   Spearman's ρ: {rho:.4f} (p={p_value:.4f})")
            
            # Store results
            all_results.append({
                "name": test_case['name'],
                "top1_match": top1_match,
                "top3_overlap": top3_overlap,
                "avg_score_diff": avg_score_diff,
                "max_score_diff": max_score_diff,
                "vllm_order": vllm_order,
                "deepinfra_order": deepinfra_order,
                "vllm_scores": vllm_scores,
                "deepinfra_scores": deepinfra_scores,
            })
        
        # Overall summary
        print(f"\n{'='*80}")
        print(f"📊 Overall Summary Across All Test Cases")
        print(f"{'='*80}")
        
        top1_matches = sum(1 for r in all_results if r['top1_match'])
        avg_top3_overlap = np.mean([r['top3_overlap'] for r in all_results])
        avg_score_diff_all = np.mean([r['avg_score_diff'] for r in all_results])
        max_score_diff_all = np.max([r['max_score_diff'] for r in all_results])
        
        print(f"\n🎯 Ranking Consistency:")
        print(f"   Top-1 matches:        {top1_matches}/{len(test_cases)} ({top1_matches/len(test_cases)*100:.1f}%)")
        print(f"   Avg Top-3 overlap:    {avg_top3_overlap:.2f}/3 ({avg_top3_overlap/3*100:.1f}%)")
        
        print(f"\n📊 Score Differences:")
        print(f"   Average score diff:   {avg_score_diff_all:.4f}")
        print(f"   Maximum score diff:   {max_score_diff_all:.4f}")
        
        print(f"\n💡 Interpretation:")
        if top1_matches >= len(test_cases) * 0.8:
            print(f"   ✅ EXCELLENT: Services agree on top result in most cases")
        elif top1_matches >= len(test_cases) * 0.6:
            print(f"   ✅ GOOD: Services show reasonable consistency")
        else:
            print(f"   ⚠️  MODERATE: Significant ranking differences between services")
        
        if avg_score_diff_all < 0.1:
            print(f"   ✅ Score differences are minimal")
        elif avg_score_diff_all < 0.2:
            print(f"   ⚠️  Score differences are moderate")
        else:
            print(f"   ⚠️  Score differences are significant")
        
        # Case-by-case summary table
        print(f"\n📋 Summary Table:")
        print(f"   {'Case':<35} {'Top-1':<8} {'Top-3':<8} {'Avg Diff':<10}")
        print(f"   {'-'*35} {'-'*8} {'-'*8} {'-'*10}")
        for r in all_results:
            top1_sym = "✅" if r['top1_match'] else "❌"
            top3_str = f"{r['top3_overlap']}/3"
            print(f"   {r['name']:<35} {top1_sym:<8} {top3_str:<8} {r['avg_score_diff']:<10.4f}")
        
        print("\n✅ Detailed rerank comparison PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Detailed comparison FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vllm_service.close()
        await deepinfra_service.close()


async def main():
    """Run all integration tests"""
    print("\n" + "🚀 " * 20)
    print("  Integration Tests for Vectorize & Rerank Services")
    print("🚀 " * 20)
    
    results = []
    
    # Test vectorize service
    result_vectorize = await test_vectorize_service()
    results.append(("Vectorize Service", result_vectorize))
    
    # Test rerank service
    result_rerank = await test_rerank_service()
    results.append(("Rerank Service", result_rerank))
    
    # Test complete pipeline
    result_pipeline = await test_retrieval_pipeline()
    results.append(("Complete Pipeline", result_pipeline))
    
    # Test comparison between vLLM and DeepInfra rerank
    result_rerank_comparison = await test_compare_vllm_deepinfra_rerank()
    results.append(("vLLM vs DeepInfra Rerank", result_rerank_comparison))
    
    # Test comparison between vLLM and DeepInfra embedding
    result_embedding_comparison = await test_compare_vllm_deepinfra_embedding()
    results.append(("vLLM vs DeepInfra Embedding", result_embedding_comparison))
    
    # Detailed rerank comparison with multiple test cases
    result_detailed_rerank = await test_detailed_rerank_comparison()
    results.append(("Detailed Rerank Comparison", result_detailed_rerank))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:40s} {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n   ⚠️  {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

