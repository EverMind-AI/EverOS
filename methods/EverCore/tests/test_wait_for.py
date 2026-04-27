import asyncio
import time


async def short_task():
    await asyncio.sleep(0.1)
    return "I am done!"


async def main():
    print(f"[{time.time():.2f}] Main started.")

    # 1. Create and wait for task1 to complete
    task1 = asyncio.create_task(short_task())
    result1 = await task1
    print(f"[{time.time():.2f}] Task 1 has already finished. Result: {result1}")
    print(f"[{time.time():.2f}] Is Task 1 done? {task1.done()}")

    # 2. Put the already completed task1 into gather
    print(f"\n[{time.time():.2f}] Now gathering the finished task.")
    start_gather_time = time.time()
    results = await asyncio.gather(task1)  # Should return immediately
    end_gather_time = time.time()

    print(f"[{time.time():.2f}] Gather completed.")
    print(f"Time taken by gather: {end_gather_time - start_gather_time:.4f} seconds.")
    print("Results:", results)


if __name__ == "__main__":
    asyncio.run(main())