"""MemSys Simple Demo - Easy to Understand!

Demonstrates how to use the memory system:
1. Store conversations
2. Search memories

Prerequisites:
    Start the API server first (in another terminal):
    uv run python src/run.py

Run the demo:
    uv run python src/bootstrap.py demo/simple_demo.py
"""

import asyncio
from demo.utils import SimpleMemoryManager


async def main():
    """Super simple usage example - just 3 steps!"""
    
    # Create memory manager
    memory = SimpleMemoryManager()
    
    memory.print_separator("🧠  MemSys Simple Demo")
    
    # ========== Step 1: Store Conversations ==========
    print("\n📝 Step 1: Store Conversations")
    memory.print_separator()
    
    await memory.store("I love playing soccer, often go to the field on weekends")
    await asyncio.sleep(2)
    
    await memory.store("Soccer is a great sport! Which team do you like?", sender="Assistant")
    await asyncio.sleep(2)
    
    await memory.store("I love Barcelona the most, Messi is my idol")
    await asyncio.sleep(2)
    
    await memory.store("I also enjoy watching basketball, NBA is my favorite")
    await asyncio.sleep(2)
    
    await memory.store("I will sleep now")
    await asyncio.sleep(2)

    await memory.store("The weather is good today")
    await asyncio.sleep(2)
    
    await memory.store("The universe is expanding")
    await asyncio.sleep(2)
    # ========== Step 2: Wait for Indexing ==========
    print("\n⏳ Step 2: Wait for Index Building")
    memory.print_separator()
    await memory.wait_for_index(seconds=10)
    
    # ========== Step 3: Search Memories ==========
    print("\n🔍 Step 3: Search Memories")
    memory.print_separator()
    
    print("\n💬 Query 1: What sports does the user like?")
    await memory.search("What sports does the user like?")
    
    print("\n💬 Query 2: What is the user's favorite team?")
    await memory.search("What is the user's favorite team?")
    
    print("\n💬 Query 3: What are the user's sports hobbies?")
    await memory.search("What are the user's sports hobbies?")
    
    # ========== Done ==========
    memory.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
