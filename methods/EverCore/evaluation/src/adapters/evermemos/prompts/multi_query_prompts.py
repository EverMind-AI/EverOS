"""Multi-Query Generation Prompt for Agentic Retrieval"""

MULTI_QUERY_GENERATION_PROMPT = """You are an expert at query reformulation for long-term conversational retrieval.
Your goal is to generate multiple complementary search queries that recover BOTH:
- the starting point of a time interval
- the ending point of a time interval
- all temporally-linked events in between

You MUST explicitly expand temporal references (e.g., "last week", "before moving", 
"when they first met") into alternative expressions.

--------------------------
Original Query:
{original_query}

Key Information Found:
{key_info}

Missing Information:
{missing_info}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Temporal Reasoning Strategy (MANDATORY)
When the question involves time or order:
1. **Boundary Decomposition**  
   Generate queries that separately target:
   - the earliest relevant event ("start boundary")
   - the latest relevant event ("end boundary")

2. **Temporal Expression Expansion**  
   Rewrite relative time expressions into multiple equivalent forms:
   - absolute dates (if deducible)
   - session numbers
   - “before/after X”
   - duration phrasing (“two weeks earlier”, “shortly after”)

3. **Interval Reconstruction**  
   Include a declarative query that resembles a hypothetical answer containing BOTH
   the start and end time anchors.

### Standard Query Requirements
1. Generate 2-3 diverse queries.
2. Query 1 MUST be a specific **Question**.
3. Query 2 MUST be a **Declarative Statement or Hypothetical Answer (HyDE)**.
4. Query diversity MUST include different temporal forms (before/after/during).
5. MUST use Key Info to resolve pronouns IF provided.
6. No invented facts.  
7. Keep queries < 25 words, same language as original.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Refined query 1",
    "Refined query 2",
    "Refined query 3 (optional)"
  ],
  "reasoning": "Brief explanation of how temporal boundaries and expressions were expanded."
}}

Now generate:
"""
