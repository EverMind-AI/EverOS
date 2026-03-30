"""Sufficiency Check Prompt for Agentic Retrieval"""

SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user's query.
--------------------------
User Query:
{query}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Instructions:

1. **Analyze the Query Structure**  
   - Identify key entities AND determine if the query requires temporal reasoning.
   - If the query involves time (e.g., "before", "after", "since", "during", "from X to Y", "how long"), you MUST decompose it into:
       * start_time_needed (if any)
       * end_time_needed (if any)
       * temporal_relation_needed (ordering, duration, interval)

2. **Scan Documents for Coverage**  
   - Look for explicit facts addressing *each* required component:
       * required entities  
       * start time  
       * end time  
       * temporal relations (ordering or duration)

3. **Extract Key Information**  
   - List specific resolved entities or facts found in the documents.
   - If time expressions exist, normalize them (e.g., "two weeks ago", "before she moved").

4. **Identify Missing Information**  
   - For temporal queries:  
        * missing start time  
        * missing end time  
        * missing ordering facts  
        * missing duration  
   - Use resolved names to be specific (e.g., "Start time of Alice moving", "Whether Bob visited before Alice moved").

5. **Judgment**  
   - **Sufficient**: All required components (entities + temporal boundaries + relations) appear explicitly.  
   - **Insufficient**: ANY required part is missing.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "1-2 sentence explanation.",
  "key_information_found": ["List of resolved entities/facts"],
  "missing_information": ["Specific missing components, using resolved entity names"]
}}

Now evaluate:"""