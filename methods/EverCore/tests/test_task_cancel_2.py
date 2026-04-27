import asyncio
import time


async def child_task():
    try:
        print(f"[{time.time():.2f}] Child task started")
        await asyncio.sleep(2)
        print(f"[{time.time():.2f}] Child task completed")
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] Child task received cancellation")
        raise


async def parent_task():
    print(f"[{time.time():.2f}] Parent task started")

    # Perform some work before await
    print(f"[{time.time():.2f}] Parent task performing some work")
    await asyncio.sleep(5)  # Simulate work

    try:
        print(f"[{time.time():.2f}] Parent task about to await child task")
        # Key point: cancellation happens here
        await child_task()
        print(f"[{time.time():.2f}] Parent task finished waiting")
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] Parent task received cancellation")
        raise


async def main():
    parent = asyncio.create_task(parent_task())

    # Cancel parent task immediately (before it reaches await)
    print(f"[{time.time():.2f}] Immediately send cancellation request")
    await asyncio.sleep(1)
    parent.cancel()

    try:
        await parent
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] Main program caught cancellation")


asyncio.run(main())