"""Chat Agent Demo - Casual Conversation Without Tool Use

Demonstrates memory extraction from casual chat (no tool calls):
1. Send one round of casual, chitchat-heavy conversation (no tool calls)
2. Wait for memory extraction (MemCell segmentation)
3. System extracts episodic memories, event logs, foresights,
   agent experiences (assistant reasoning patterns), and agent skills

The mock conversation simulates a friendly chat assistant handling:
- Weekend plans, favorite movies, and cooking hobby

Even without tool calls, the system extracts agent experiences from the
assistant's conversational strategies and may derive agent skills from them.

Prerequisites:
    Start the API server first (in another terminal):
    uv run python src/run.py

Run the demo:
    uv run python src/bootstrap.py demo/chat_agent_demo.py
"""

import asyncio

from demo.utils.agent_demo_helpers import (
    AgentDemoRunner,
    MEMORY_TYPE_PRINTERS,
    print_separator,
    print_search_case_results,
    print_search_skill_results,
)


# ==================== Mock Chat Conversations ====================

def build_first_conversation():
    """Build a casual chat conversation - Round 1.

    Topics: weekend plans, favorite movies, cooking hobby.
    No tool calls at all - pure assistant-user chitchat.
    """
    messages = [
        # ===== Topic 1: Weekend plans =====
        {
            "role": "user",
            "content": "Hey! What should I do this weekend? I'm so bored staying at home.",
        },
        {
            "role": "assistant",
            "content": (
                "Hey there! Here are some ideas depending on your mood:\n\n"
                "- **Outdoor**: Go for a hike, visit a local farmers market, or try a new park\n"
                "- **Creative**: Pick up a new recipe, start a small DIY project, or try sketching\n"
                "- **Social**: Invite friends for a board game night or check out a local event\n"
                "- **Chill**: Binge a new series, visit a bookstore, or just take a long walk\n\n"
                "What kind of vibe are you going for?"
            ),
        },
        {
            "role": "user",
            "content": (
                "I think I'll go hiking! There's a trail near Mt. Tamalpais I've been wanting "
                "to try. My friend Sarah mentioned it has amazing views of the Golden Gate Bridge. "
                "We're planning to go Saturday morning around 7am."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Mt. Tam is gorgeous! The Dipsea Trail or the Matt Davis Trail both have "
                "incredible views. Since you're going at 7am, you'll probably catch the sunrise "
                "fog rolling over the bridge - it's magical.\n\n"
                "A few tips:\n"
                "- Bring layers, mornings can be chilly even in summer\n"
                "- Pack extra water - some trails are longer than expected\n"
                "- The parking lot fills up fast on weekends, so 7am is smart\n\n"
                "Have a great time with Sarah! Take some photos of the bridge view."
            ),
        },

        # ===== Topic 2: Favorite movies =====
        {
            "role": "user",
            "content": (
                "By the way, I watched Interstellar again last night. It's still my favorite "
                "movie of all time. I cry every time at the bookshelf scene. Have you seen it?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Interstellar is a masterpiece! The bookshelf scene where Cooper realizes he's "
                "been the 'ghost' all along is incredibly emotional. Hans Zimmer's organ score "
                "makes it even more powerful.\n\n"
                "If you love Interstellar, you might also enjoy:\n"
                "- **Arrival** - Another cerebral sci-fi film about time and communication\n"
                "- **Contact** - Also about reaching across the cosmos, with Jodie Foster\n"
                "- **The Prestige** - Another Nolan film with mind-bending storytelling\n\n"
                "What is it about Interstellar that resonates with you the most?"
            ),
        },
        {
            "role": "user",
            "content": (
                "It's the father-daughter relationship. I have a 5-year-old daughter named Emma, "
                "and the idea of missing years of her life just destroys me. I've actually "
                "already seen Arrival - loved it! The twist about the language changing how you "
                "perceive time was brilliant. I'll check out Contact though, thanks!"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "That makes total sense - the parent-child theme in Interstellar hits differently "
                "when you have kids. Cooper watching those video messages of Murph growing up "
                "while he's been gone... that scene is devastating.\n\n"
                "With a 5-year-old, you've got the perfect age where every moment feels precious. "
                "Emma is lucky to have a dad who appreciates that.\n\n"
                "You'll love Contact - it's more grounded than Interstellar but has a similar "
                "emotional core about human connection. Matthew McConaughey is actually in it too, "
                "which is a fun coincidence!"
            ),
        },

        # ===== Topic 3: Cooking hobby =====
        {
            "role": "user",
            "content": (
                "Oh, I also wanted to mention - I've been really getting into cooking lately. "
                "Last week I made homemade ramen from scratch, including the broth. Took 12 hours "
                "but it was totally worth it. My wife Lisa said it was better than our favorite "
                "ramen shop!"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "12 hours of broth simmering? That's serious dedication! Homemade ramen is one of "
                "those dishes where the effort really shows in the final product. The depth of "
                "flavor from a long-simmered tonkotsu or shoyu broth is hard to replicate with "
                "shortcuts.\n\n"
                "What kind of broth did you make? Tonkotsu (pork bone), shoyu (soy sauce based), "
                "or miso? And did you make your own noodles too, or just the broth?\n\n"
                "Also, huge compliment from Lisa - when your spouse says it's better than "
                "restaurant-quality, you know you've nailed it!"
            ),
        },
        {
            "role": "user",
            "content": (
                "Tonkotsu! I boiled pork bones for about 10 hours to get that creamy white broth. "
                "I didn't make noodles from scratch though - used Sun Noodle brand, they're "
                "basically restaurant quality. Next time I want to try making the noodles too. "
                "I've been watching a lot of cooking videos on YouTube, especially from a channel "
                "called 'Way of Ramen'. Their techniques are incredible."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Tonkotsu is the ultimate test of patience! Getting that milky white emulsion "
                "from pork bones is all about maintaining a rolling boil - sounds like you "
                "nailed it.\n\n"
                "Sun Noodle is a great choice - they literally supply noodles to top ramen shops "
                "across the US. Smart move to nail the broth first and tackle noodles next.\n\n"
                "Making fresh alkaline noodles is a fun project. You'll need kansui (alkaline "
                "mineral water) to get that springy, yellow texture. The dough is quite stiff so "
                "a pasta machine helps a lot.\n\n"
                "Way of Ramen is excellent! Their tonkotsu episodes are super detailed. If you "
                "want to level up, also check out 'Alex' on the French Cooking Academy channel - "
                "he has some great Asian noodle technique videos too."
            ),
        },
    ]
    return messages


# ==================== Main ====================

async def main():
    runner = AgentDemoRunner(
        group_id_prefix="chat_agent_demo",
        group_name="Chat Agent Demo Session",
        description="Chat Agent Memory Demo - Casual Conversation",
        tags=["demo", "agent", "chat"],
        msg_prefix="chat_msg",
    )

    print_separator("Chat Agent Memory Demo - Casual Conversation (No Tool Use)")

    # Step 1: Initialize conversation metadata
    print("\nStep 1: Initialize conversation metadata")
    print_separator()
    await runner.save_conversation_meta()

    # Step 2: Send first casual conversation
    print("\nStep 2: Send first conversation round (weekend, movies, cooking)")
    print_separator()
    print("  Pure chitchat - no tool calls.\n")

    messages1 = build_first_conversation()
    print(f"  Sending {len(messages1)} messages (3 topics)...\n")

    for idx, msg in enumerate(messages1):
        is_last = (idx == len(messages1) - 1)
        await runner.send_agent_message(msg, idx, flush=is_last)
        await asyncio.sleep(0.3)

    # Step 3: Wait for extraction
    print("\nStep 3: Wait for extraction pipeline")
    print_separator()
    print("  Waiting 60 seconds for memory extraction...")
    await asyncio.sleep(60)

    # Step 4: Fetch and print all extracted memories
    print("\nStep 4: Fetch all extracted memories")
    print("  (Expect: all memory types populated, including agent_case & agent_skill)\n")

    for memory_type, label, printer in MEMORY_TYPE_PRINTERS:
        print(f"\n--- {label} ---")
        memories = await runner.fetch_memories(memory_type)
        print(f"  ({len(memories)} records)")
        printer(memories)

    # Step 5: Search demo - verify episodic/event_log are searchable
    print("\nStep 5: Search memories")

    print("\n--- Hybrid Search: Episodic Memory ---")
    print('  Query: "weekend hiking plans with friends"')
    hits = await runner.search_memories(
        "weekend hiking plans with friends", "episodic_memory"
    )
    print(f"  ({len(hits)} hits)")
    for i, h in enumerate(hits[:3], 1):
        score = h.get("score", 0.0)
        summary = h.get("summary") or h.get("episode") or "N/A"
        print(f"\n  [{i}] score={score:.4f}")
        print(f"      {summary[:150]}")

    # Agent Memory searches (expect 0 - chat demo has no agent trajectories/skills)
    agent_memory_queries = [
        "cooking ramen from scratch",
        "weekend outdoor activities with friends",
        "favorite sci-fi movies like Interstellar",
        "how to make homemade ramen",
        "plan a hiking trip for beginners",
    ]
    for q in agent_memory_queries:
        print(f"\n--- Hybrid Search: Agent Memory ---")
        print(f'  Query: "{q}"')
        result = await runner.search_memories(q, "agent_memory")
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    print_separator("Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
