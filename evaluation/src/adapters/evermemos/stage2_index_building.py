import json
import os
import sys
import pickle
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import asyncio


from evaluation.src.adapters.evermemos.config import ExperimentConfig
from agentic_layer.vectorize_service import get_vectorize_service


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)

    # Verify stopwords availability
    try:
        from nltk.corpus import stopwords

        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


def build_searchable_text(doc: dict) -> str:
    """
    Build searchable text from a document with weighted fields.

    Priority:
    1. If event_log exists, use atomic_fact for indexing
    2. Otherwise, fall back to original fields:
       - "subject" corresponds to "title" (weight * 3)
       - "summary" corresponds to "summary" (weight * 2)
       - "episode" corresponds to "content" (weight * 1)
    """
    parts = []

    # Prefer event_log's atomic_fact (if exists)
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            # Handle nested atomic_fact structure
            # atomic_fact can be list of strings or list of dicts (containing "fact" and "embedding")
            for fact in atomic_facts:
                if isinstance(fact, dict) and "fact" in fact:
                    # New format: {"fact": "...", "embedding": [...]}
                    parts.append(fact["fact"])
                elif isinstance(fact, str):
                    # Old format: pure string list (backward compatible)
                    parts.append(fact)
            return " ".join(str(fact) for fact in parts if fact)

    # Fall back to original fields (maintain backward compatibility)
    # Title has highest weight (repeat 3 times)
    if doc.get("subject"):
        parts.extend([doc["subject"]] * 3)

    # Summary (repeat 2 times)
    if doc.get("summary"):
        parts.extend([doc["summary"]] * 2)

    # Content
    if doc.get("episode"):
        parts.append(doc["episode"])

    return " ".join(str(part) for part in parts if part)


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]

    return processed_tokens


def build_bm25_index(
    config: ExperimentConfig, data_dir: Path, bm25_save_dir: Path
) -> list[list[float]]:
    # --- NLTK Setup ---
    print("Ensuring NLTK data is available...")
    ensure_nltk_data()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # --- Data Loading and Processing ---
    # corpus = [] # This line is removed as per the new_code
    # original_docs = [] # This line is removed as per the new_code

    print(f"Reading data from: {data_dir}")

    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        print(f"\nProcessing {file_path.name}...")

        corpus = []
        original_docs = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for doc in data:
                original_docs.append(doc)
                searchable_text = build_searchable_text(doc)
                tokenized_text = tokenize(searchable_text, stemmer, stop_words)
                corpus.append(tokenized_text)

        if not corpus:
            print(
                f"Warning: No documents found in {file_path.name}. Skipping index creation."
            )
            continue

        print(f"Processed {len(original_docs)} documents from {file_path.name}.")

        # --- BM25 Indexing ---
        print(f"Building BM25 index for {file_path.name}...")
        bm25 = BM25Okapi(corpus)

        # --- Saving the Index ---
        index_data = {"bm25": bm25, "docs": original_docs}

        output_path = bm25_save_dir / f"bm25_index_conv_{i}.pkl"
        print(f"Saving index to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)


async def build_emb_index(config: ExperimentConfig, data_dir: Path, emb_save_dir: Path):
    """
    Build Embedding index (stable version).

    Performance optimization strategy:
    1. Controlled concurrency: strictly follow API Semaphore(5) limit
    2. Conservative batch size: 256 texts/batch (avoid timeouts)
    3. Serial batch submission: grouped submission to avoid queue buildup
    4. Progress monitoring: real-time progress and speed display

    Optimization effects:
    - Stability first, avoid timeouts and API overload
    - API concurrency: 5 (controlled by vectorize_service.Semaphore)
    - Batch size: 256 (balance stability and efficiency)
    """
    # Conservative batch size (avoid timeouts)
    BATCH_SIZE = (
        256  # Use larger batches (single API call processes more, reduce request count)
    )
    MAX_CONCURRENT_BATCHES = 5  # Strictly control concurrency (match Semaphore(5))

    import time  # For performance statistics

    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {file_path.name} for embedding...")
        print(f"{'='*60}")

        with open(file_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        texts_to_embed = []
        doc_field_map = []
        for doc_idx, doc in enumerate(original_docs):
            # Prefer event_log (if exists)
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                atomic_facts = doc["event_log"]["atomic_fact"]
                if isinstance(atomic_facts, list) and atomic_facts:
                    # calculate embedding for each atomic_fact separately (MaxSim strategy)
                    # This precisely matches specific atomic facts, avoiding semantic dilution
                    for fact_idx, fact in enumerate(atomic_facts):
                        # compatible with both formats (string / dict)
                        fact_text = None
                        if isinstance(fact, dict) and "fact" in fact:
                            # New format: {"fact": "...", "embedding": [...]}
                            fact_text = fact["fact"]
                        elif isinstance(fact, str):
                            # Old format: pure string
                            fact_text = fact

                        # Ensure fact is non-empty
                        if fact_text and fact_text.strip():
                            texts_to_embed.append(fact_text)
                            doc_field_map.append((doc_idx, f"atomic_fact_{fact_idx}"))
                    continue

            # Fall back to original fields (maintain backward compatibility)
            for field in ["subject", "summary", "episode"]:
                if text := doc.get(field):
                    texts_to_embed.append(text)
                    doc_field_map.append((doc_idx, field))

        if not texts_to_embed:
            print(
                f"Warning: No documents found in {file_path.name}. Skipping embedding creation."
            )
            continue

        total_texts = len(texts_to_embed)
        total_batches = (total_texts + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total texts to embed: {total_texts}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Total batches: {total_batches}")
        print(f"Max concurrent batches: {MAX_CONCURRENT_BATCHES}")
        print(f"\nStarting parallel embedding generation...")

        # Stable batch processing (avoid timeouts)
        start_time = time.time()

        async def process_batch_with_retry(
            batch_idx: int, batch_texts: list, max_retries: int = 3
        ) -> tuple[int, list]:
            """Process single batch (async + retry)."""
            for attempt in range(max_retries):
                try:
                    # Call API to get embeddings (concurrency controlled by Semaphore(5))
                    batch_embeddings = await get_vectorize_service().get_embeddings(
                        batch_texts
                    )
                    return (batch_idx, batch_embeddings)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2.0 * (2**attempt)  # Exponential backoff: 2s, 4s
                        print(
                            f"  ⚠️  Batch {batch_idx + 1}/{total_batches} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        print(
                            f"  ❌ Batch {batch_idx + 1}/{total_batches} failed after {max_retries} attempts: {e}"
                        )
                        return (batch_idx, [])

        # Grouped serial submission (avoid queue buildup causing timeouts)
        print(
            f"Processing {total_batches} batches in groups of {MAX_CONCURRENT_BATCHES}..."
        )

        batch_results = []
        completed = 0

        # Grouped submission, max MAX_CONCURRENT_BATCHES concurrent per group
        for group_start in range(0, total_texts, BATCH_SIZE * MAX_CONCURRENT_BATCHES):
            # Calculate batch range for current group
            group_end = min(
                group_start + BATCH_SIZE * MAX_CONCURRENT_BATCHES, total_texts
            )
            group_tasks = []

            for j in range(group_start, group_end, BATCH_SIZE):
                batch_idx = j // BATCH_SIZE
                batch_texts = texts_to_embed[j : j + BATCH_SIZE]
                task = process_batch_with_retry(batch_idx, batch_texts)
                group_tasks.append(task)

            # Process current group concurrently (max MAX_CONCURRENT_BATCHES)
            print(
                f"  Group {group_start//BATCH_SIZE//MAX_CONCURRENT_BATCHES + 1}: Processing {len(group_tasks)} batches concurrently..."
            )
            group_results = await asyncio.gather(*group_tasks, return_exceptions=False)
            batch_results.extend(group_results)

            completed += len(group_tasks)
            progress = (completed / total_batches) * 100
            print(f"  Progress: {completed}/{total_batches} batches ({progress:.1f}%)")

            # Inter-group delay (give API server breathing room)
            if group_end < total_texts:
                await asyncio.sleep(1.0)  # 1s inter-group delay

        # Reorganize results by batch order
        all_embeddings = []
        for batch_idx, batch_embeddings in sorted(batch_results, key=lambda x: x[0]):
            all_embeddings.extend(batch_embeddings)

        elapsed_time = time.time() - start_time
        speed = total_texts / elapsed_time if elapsed_time > 0 else 0
        print(f"\n✅ Embedding generation complete!")
        print(f"   - Total texts: {total_texts}")
        print(f"   - Total embeddings: {len(all_embeddings)}")
        print(f"   - Time elapsed: {elapsed_time:.2f}s")
        print(f"   - Speed: {speed:.1f} texts/sec")
        print(f"   - Average batch time: {elapsed_time/total_batches:.2f}s")

        # Verify result completeness
        if len(all_embeddings) != total_texts:
            print(
                f"   ⚠️  Warning: Expected {total_texts} embeddings, got {len(all_embeddings)}"
            )
        else:
            print(f"   ✓ All embeddings generated successfully")

        # Re-associate embeddings with their original documents and fields
        # Support multiple atomic_fact embeddings per document (for MaxSim strategy)
        doc_embeddings = [{"doc": doc, "embeddings": {}} for doc in original_docs]

        for (doc_idx, field), emb in zip(doc_field_map, all_embeddings):
            # If atomic_fact field, save as list (support multiple atomic_facts)
            if field.startswith("atomic_fact_"):
                if "atomic_facts" not in doc_embeddings[doc_idx]["embeddings"]:
                    doc_embeddings[doc_idx]["embeddings"]["atomic_facts"] = []
                doc_embeddings[doc_idx]["embeddings"]["atomic_facts"].append(emb)
            else:
                # Save other fields directly
                doc_embeddings[doc_idx]["embeddings"][field] = emb

        # The final structure of the saved .pkl file will be a list of dicts:
        # [
        #     {
        #         "doc": { ... original document ... },
        #         "embeddings": {
        #             "atomic_facts": [  # New: atomic_fact embeddings list (for MaxSim)
        #                 [ ... embedding vector for fact 0 ... ],
        #                 [ ... embedding vector for fact 1 ... ],
        #                 ...
        #             ],
        #             "subject": [ ... embedding vector ... ],  # Backward compatible legacy fields
        #             "summary": [ ... embedding vector ... ],
        #             "episode": [ ... embedding vector ... ]
        #         }
        #     },
        #     ...
        # ]
        output_path = emb_save_dir / f"embedding_index_conv_{i}.pkl"
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(doc_embeddings, f)


async def main():
    """Main function to build and save the BM25 index."""
    # --- Configuration ---
    # The directory containing the JSON files
    config = ExperimentConfig()
    data_dir = Path(__file__).parent / config.experiment_name / "memcells"
    bm25_save_dir = Path(__file__).parent / config.experiment_name / "bm25_index"
    emb_save_dir = Path(__file__).parent / config.experiment_name / "vectors"
    os.makedirs(bm25_save_dir, exist_ok=True)
    os.makedirs(emb_save_dir, exist_ok=True)
    build_bm25_index(config, data_dir, bm25_save_dir)
    if config.use_emb:
        await build_emb_index(config, data_dir, emb_save_dir)
    # data_dir = Path("/Users/admin/Documents/Projects/b001-memsys/evaluation/locomo_evaluation/results/locomo_evaluation_0/")

    # Where to save the final index file
    # output_path = data_dir / "bm25_index.pkl" # This line is removed as per the new_code

    print("\nAll indexing complete!")


if __name__ == "__main__":
    asyncio.run(main())
