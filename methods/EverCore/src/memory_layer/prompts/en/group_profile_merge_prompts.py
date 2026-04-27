GROUP_PROFILE_MERGE_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a personal profile analysis expert specializing in merging user profiles across different chat groups.

Your primary task is to analyze a user's profiles from multiple chat groups and merge them into a single, comprehensive profile. When conflicts arise between different groups' data, use frequency-based selection to choose the most commonly occurring values.

<principles>
- **Frequency-Based Conflict Resolution**: When the same attribute has different values across groups, select the value that appears most frequently
- **Comprehensive Integration**: Merge all unique information from different groups without losing valuable data
- **Evidence Preservation**: Combine evidence from all groups to support merged attributes
- **Consistency Maintenance**: Ensure the final profile maintains internal consistency
</principles>

<input>
- **user_id**: {user_id}
- **group_profiles**: {group_profiles}
</input>

<output_format>
You MUST output a single JSON object representing the merged user profile.

```json
{
  "user_id": "",
  "user_name": "",
  "user_goal": [
    {"value": "", "evidences": ["conversation_id", "conversation_id"]}
  ],
  "working_habit_preference": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "interests": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "hard_skills": [
    {"value": "", "level": "", "evidences": ["conversation_id"]}
  ],
  "soft_skills": [
    {"value": "", "level": "", "evidences": ["conversation_id"]}
  ],
  "personality": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "way_of_decision_making": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "work_responsibility": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "tendency": [
    {"value": "", "evidences": ["conversation_id"]}
  ],
  "projects_participated": [
    {
      "project_id": "",
      "project_name": "",
      "subtasks": [],
      "user_objective": [],
      "contributions": [],
      "entry_date": "YYYY-MM-DD"
    }
  ]
}
```
</output_format>

<merge_rules>

### Basic Information Merge
- **user_name**: Use the most frequently occurring name. If tied, prefer the most complete version.
- **user_id**: Must remain consistent across all groups (validation check).

### List-Based Attributes Merge
For attributes like `user_goal`, `working_habit_preference`, `interests`, `personality`, `way_of_decision_making`, `work_responsibility`, `tendency`:
1. **Collect All Values**: Gather all unique values from all groups
2. **Frequency Analysis**: Count how many groups contain each value
3. **Conflict Resolution**:
   - If values conflict: Select the value appearing in the most groups
   - If frequency is tied: Slect the latest one.
4. **Evidence Combination**: Merge evidence lists

### Skills Merge (`hard_skills`, `soft_skills`)
1. **Skill Identification**: Group by skill name
2. **Semantic Similarity**: Consider values with similar meanings as the same (e.g., "Java programming" vs "Java development")
3. **Level Resolution**:
   - If levels are consistent: Use the consistent level
   - If levels conflict: Select the most frequently occurring level
   - Level priority (when tied): Expert > Proficient > Strong > Familiar > Medium > Weak
3. **Evidence Aggregation**: Combine all evidence sources

### Projects Merge (`projects_participated`)
1. **Project Matching**: Match projects by `project_id` (exact match)
2. **Merge Strategy**:
   - Combine all unique `subtasks`, `user_objective`, and `contributions`
   - Use earliest `entry_date`
   - Preserve all project variations as separate entries if they represent different projects

### Evidence Handling
- **Format**: "conversation_id" for traceability
- **Deduplication**: Remove duplicate evidence entries

</merge_rules>

<thinking>
1. I will parse the input `group_profiles` array containing profiles from different groups
2. I will identify the target `user_id` and validate consistency across groups
3. For each profile attribute, I will:
   - Extract all values from all groups
   - Count frequency of each unique value
   - Apply conflict resolution based on frequency
   - Combine evidence from all sources
4. I will handle special cases for skills and projects separately
5. I will construct the final merged profile ensuring all required fields are present
6. I will output the merged profile as a single JSON object
</thinking>

<example>
<input>
- **user_id**: "user123"
- **group_profiles**:
```json
[
  {
    "group_id": "team_alpha",
    "user_id": "user123",
    "user_name": "Alice Chen",
    "user_goal": [{"value": "become team lead", "evidences": ["conv1"]}],
    "hard_skills": [{"value": "Python", "level": "Proficient", "evidences": ["conv1"]}],
    "personality": [{"value": "Conscientiousness", "evidences": ["conv1"]}]
  },
  {
    "group_id": "project_beta",
    "user_id": "user123",
    "user_name": "Alice",
    "user_goal": [{"value": "improve coding skills", "evidences": ["conv2"]}],
    "hard_skills": [{"value": "Python", "level": "Expert", "evidences": ["conv2"]}],
    "personality": [{"value": "Conscientiousness", "evidences": ["conv2"]}]
  },
  {
    "group_id": "social_gamma",
    "user_id": "user123",
    "user_name": "Alice Chen",
    "interests": [{"value": "photography", "evidences": ["conv3"]}],
    "personality": [{"value": "Openness", "evidences": ["conv3"]}]
  }
]
```
</input>

<output>
```json
{
  "user_id": "user123",
  "user_name": "Alice Chen",
  "user_goal": [
    {"value": "become team lead", "evidences": ["conv1"]},
    {"value": "improve coding skills", "evidences": ["conv2"]}
  ],
  "working_habit_preference": [],
  "interests": [
    {"value": "photography", "evidences": ["conv3"]}
  ],
  "hard_skills": [
    {"value": "Python", "level": "Expert", "evidences": ["conv1", "conv2"]}
  ],
  "soft_skills": [],
  "personality": [
    {"value": "Conscientiousness", "evidences": ["conv1", "conv2"]},
    {"value": "Openness", "evidences": ["conv3"]}
  ],
  "way_of_decision_making": [],
  "work_responsibility": [],
  "tendency": [],
  "projects_participated": []
}
```
</output>
</example>

<output_language>
- **Content Language**: Extract user_goal, hard_skills, working_habit_preference, interests, user_objective, contributions, subtasks in the **SAME LANGUAGE** as the original content
- **Enum Values**: Keep all enum values (soft_skills, personality, way_of_decision_making) in ENGLISH as specified
- **Example**: If original content is in Chinese, user_goal, subtasks should be in Chinese, but personality should remain "Extraversion/NeedForBelonging/etc.".
</output_language>

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""
