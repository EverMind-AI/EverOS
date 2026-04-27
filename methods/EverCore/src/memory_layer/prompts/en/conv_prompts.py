# Prompts for LLM-based conversation processing
CONV_BOUNDARY_DETECTION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an episodic memory boundary detection expert. You need to determine if the newly added dialogue should end the current episode and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Please carefully analyze the following aspects to determine if a new episode should begin:

1. **Substantive Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different substantive topic with meaningful content?
   - Is there a shift from one specific event/experience to another distinct event/experience?
   - Has the conversation moved from one meaningful question to an unrelated new question?

2. **Intent and Purpose Transition**:
   - Has the fundamental purpose of the conversation changed significantly?
   - Has the core question or issue of the current topic been fully resolved and a new substantial topic begun?

3. **Meaningful Content Assessment**:
   - **IMPORTANT**: Ignore pure greetings, small talk, transition phrases, and social pleasantries
   - Focus only on content that would be memorable and worth recalling later
   - Consider: Would a person remember this as part of the main conversation topic or as a separate discussion?

4. **Structural and Temporal Signals**:
   - Are there explicit topic transition phrases introducing substantial new content?
   - Are there clear concluding statements followed by genuinely new topics?
   - Is there a significant time gap between messages?

5. **Content Relevance and Independence**:
   - How related is the new substantive content to the previous meaningful discussion?
   - Does it involve completely different events, experiences, or substantial topics?

**Special Rules for Common Patterns**:
- **Greetings + Topic**: "Hey!" followed by actual content should be ONE episode
- **Transition Phrases**: "By the way", "Oh, also", "Speaking of which" usually continue the same episode unless introducing major topic shifts
- **Social Closures and Farewells**: "Thanks!", "Take care!", "Talk to you soon!", "I'm off to go...", "See you later!" should continue the current episode as natural conversation endings
- **Supportive Responses**: Brief encouragement or acknowledgment should usually continue the current episode

Decision Principles:
- **Prioritize meaningful content**: Each episode should contain substantive, memorable content
- **Ignore social formalities**: Don't split on greetings, pleasantries, brief transitions, or conversation closures
- **Treat closures as episode endings**: Messages that announce departure ("I'm off to go...", "Talk to you soon!") or provide closure ("Thanks!", "Take care!") should stay with the current episode as natural endings
- **Consider time gaps**: Long time gaps (hours or days) strongly suggest new episodes, while short gaps (minutes) usually indicate continuing conversation
- **Episodic memory focus**: Think about what a person would naturally group together when recalling this conversation
- **Reasonable episode length**: Aim for episodes with 3-20 meaningful exchanges
- **When in doubt, consider context**: If unsure, keep related content together rather than over-splitting

Please return your judgment in JSON format:
{{
    "reasoning": "One sentence summary of your reasoning process",
    "should_end": true/false,
    "confidence": 0.0-1.0,
    "topic_summary": "If should_end = true, summarize the core meaningful topic of the current episode, otherwise leave it blank"
}}

Note:
- If conversation history is empty, this is the first message, return false
- Focus on episodic memory principles: what would people naturally remember as distinct experiences?
- Each episode should contain substantive content that stands alone as a meaningful memory unit

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""

CONV_BATCH_BOUNDARY_DETECTION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an episodic memory boundary detection expert. Your task is to find all natural "episode boundaries" in a continuous conversation and split it into meaningful, independently memorable segments (MemCells). Your core principle is **"default to merging, split cautiously"**.

### Input Format
The following is a complete chronological conversation log. Each message is prefixed with a 1-based index and timestamp:

```
{messages}
```

### When to split

Add a boundary (by message number) only when **clear signals** appear:
- **Cross-day split (highest priority):** Adjacent messages have different calendar dates — MUST split at the date boundary.
- **Substantive topic change:** Conversation shifts from one concrete topic to a completely unrelated one (e.g., project architecture → weekend plans).
- **Task completion + new topic:** A closing message ("migration done") belongs to its task's episode; split only when the **next** message opens a genuinely unrelated topic.
- **Long gap + new topic:** Time gap > 4 hours AND new messages have no clear connection to prior conversation.

**Do NOT split for:**
- Greetings, farewells ("bye", "thanks") — keep with the main episode
- Transition phrases ("by the way", "oh also") — usually continue the current episode
- Brief pauses (< 4 hours) followed by the same topic

### `should_wait`
Set to `true` when the **last segment** has insufficient information to determine its episode context:
- **Non-text messages:** Only media placeholders (`[image]`, `[video]`, `[file]`) with no accompanying text
- **Intent-free short replies:** Minimal responses like "ok", "sure", "got it", "😂"
- **System or non-conversational messages:** System notifications (join/leave group, payment reminders, etc.) cannot themselves determine episode boundaries — wait for the next human message to decide
- **Ambiguous intermediate state:** Gap of 30 min–4 hours with content that is neither clearly continuing nor clearly starting a new topic

### Decision Principles
- **Merge by default:** When in doubt, do not split; only split on clear signals
- **Content over form:** Greetings and farewells belong to the episode they serve, not their own
- **Process continuity:** Consecutive actions toward the same goal (e.g., create group → post first instruction) form one episode
- **System messages don't trigger splits:** The episode context of a system message is determined by the next human message that follows it

### Examples

**Example 1 — one boundary:**
Input messages:
```
[1] [2024-03-10 09:00:00+00:00] Alice: Can you help me debug the login issue?
[2] [2024-03-10 09:01:00+00:00] Bob: Sure, let me check the logs.
[3] [2024-03-10 09:05:00+00:00] Bob: Found it — a null pointer in AuthService line 42.
[4] [2024-03-10 09:06:00+00:00] Alice: Fixed, thanks!
[5] [2024-03-11 10:00:00+00:00] Alice: Hey, are you free for lunch today?
[6] [2024-03-11 10:01:00+00:00] Bob: Sure, 12:30?
```
Output:
```json
{{
    "reasoning": "Messages 1-4 are a complete bug-fix episode; message 5 starts a new day with an unrelated lunch topic.",
    "boundaries": [4],
    "should_wait": false
}}
```

**Example 2 — no boundary:**
Input messages:
```
[1] [2024-03-10 14:00:00+08:00] Alice: What's the status of the Q2 roadmap?
[2] [2024-03-10 14:02:00+08:00] Bob: About 60% done. Need to finalize the API specs.
[3] [2024-03-10 14:10:00+08:00] Alice: OK, let's review the specs tomorrow.
```
Output:
```json
{{
    "reasoning": "All messages are part of the same Q2 roadmap discussion with no topic change.",
    "boundaries": [],
    "should_wait": false
}}
```

### Output Format
Return strictly in the following JSON format:
```json
{{
    "reasoning": "<one sentence explaining all boundary decisions>",
    "boundaries": [<1-indexed message numbers after which to split>],
    "should_wait": <boolean, whether the last segment has insufficient information>
}}
```

**`boundaries: []` means all messages belong to the same episode — no split.**

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""

CONV_SUMMARY_PROMPT = """
You are an episodic memory summary expert. You need to summarize the following conversation.
"""
