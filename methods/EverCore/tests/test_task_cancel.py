import asyncio
import time


async def child_task(name, swallow_cancel=False, hang=False):
    try:
        print(f"[{name}] Child task starts running")
        await asyncio.sleep(5)  # Simulate long-running work
        print(f"[{name}] Child task completes normally")
    except asyncio.CancelledError:
        print(f"[{name}] Child task receives cancellation request")
        if hang:
            print(f"[{name}] Child task intentionally hangs and does not exit")
            while True:  # Simulate hanging
                print(f"[{name}] Child task is stuck and hanging")
                await asyncio.sleep(1)
        elif swallow_cancel:
            print(f"[{name}] Child task swallows the cancellation request")
        else:
            print(f"[{name}] Child task rethrows the cancellation request")
            raise  # Re-raise the exception

    print(f"[{name}] Child task finally ends")


async def parent_task(child_name, swallow=False, hang=False):
    try:
        print(f"[Parent task] Starts running, creating child task {child_name}")
        child = asyncio.create_task(child_task(child_name, swallow, hang))

        # Simulate other work in parent task
        await asyncio.sleep(2)
        print(f"[Parent task] Waiting for child task to complete")
        await child
        print(f"[Parent task] Completes normally")
    except asyncio.CancelledError:
        print(f"[Parent task] Receives cancellation request")
        raise  # Re-raise the exception
    print(f"[Parent task] Finally ends")


async def main():
    # Scenario 1: Normal cancellation propagation
    print("\n===== Scenario 1: Normal cancellation propagation =====")
    task1 = asyncio.create_task(parent_task("Normal child task"))
    await asyncio.sleep(1)
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        print("[Main] Parent task has been cancelled")
    await asyncio.sleep(5)
    # Scenario 2: Child task swallows cancellation
    print("\n===== Scenario 2: Child task swallows cancellation =====")
    task2 = asyncio.create_task(parent_task("Swallowing child task", swallow=True))
    await asyncio.sleep(1)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        print("[Main] Parent task has been cancelled")

    await asyncio.sleep(5)
    # Scenario 3: Child task hangs
    print("\n===== Scenario 3: Child task hangs =====")
    task3 = asyncio.create_task(parent_task("Hanging child task", hang=True))
    await asyncio.sleep(1)
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        print("[Main] Parent task has been cancelled")
    print("[Main] Finally ends")


if __name__ == "__main__":
    asyncio.run(main())