"""Refined Query Generation Prompt for Agentic Retrieval"""

REFINED_QUERY_PROMPT = """You are an expert at query reformulation for information retrieval.

**Task**: Generate a refined query that targets the missing information in the retrieved results.

**Original Query**:
{original_query}

**Retrieved Documents** (insufficient):
{retrieved_docs}

**Missing Information**:
{missing_info}

**Instructions**:
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum).

**Examples**:

Example 1:
Original Query: "What does Alice like?"
Missing Info: ["Alice's specific interests or hobbies"]
Refined Query: "What are Alice's hobbies and interests?"

Example 2:
Original Query: "Tell me about the meeting"
Missing Info: ["meeting date", "location", "participants"]
Refined Query: "When and where was the meeting held, and who attended?"

Example 3:
Original Query: "Bob's project"
Missing Info: ["project name", "status", "purpose"]
Refined Query: "What is the name, current status, and purpose of Bob's project?"

Now generate the refined query (output only the refined query, no additional text):
Original Query: {original_query}
Missing Info: {missing_info}

Refined Query:
"""

