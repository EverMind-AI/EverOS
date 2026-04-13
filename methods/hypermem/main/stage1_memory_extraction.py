"""
Stage 1: Memory Extraction (Episode Boundary Detection)

Reads raw LoCoMo conversation data, detects episode boundaries using LLM,
generates episodic memory for each episode, and saves structured Episode files.

Data flow:
raw conversations → boundary detection → episode memory generation → save
"""

import json
import sys
import uuid
import asyncio
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hypermem.config import ExperimentConfig
from hypermem.llm.llm_provider import LLMProvider
from hypermem.types import RawDataType, Episode
from hypermem.extractors.episode_extractor import RawData
from hypermem.extractors.conv_episode_extractor import (
    ConvEpisodeExtractor, ConvEpisodeExtractRequest
)
from hypermem.utils.datetime_utils import to_iso_format, from_iso_format, get_now_with_timezone

console = Console()


# ==================== Data Loading ====================

def parse_locomo_timestamp(timestamp_str: str) -> datetime:
    """Parse LoCoMo timestamp format (e.g. '3:00 PM on 14 March, 2024') to datetime."""
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")


def load_locomo_raw_data(locomo_data_path: str) -> Dict[str, list]:
    """
    Load LoCoMo dataset and convert to message lists per conversation.

    Returns:
        Dict mapping conversation ID to list of message dicts
    """
    with open(locomo_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_data_dict = {}
    conversations = [data[i]['conversation'] for i in range(len(data))]
    print(f"   [INFO] Found {len(conversations)} conversations")

    for con_id, conversation in enumerate(conversations):
        messages = []
        session_keys = sorted(
            [key for key in conversation
             if key.startswith("session_") and not key.endswith("_date_time")]
        )

        print(f"   [INFO] Found {len(session_keys)} sessions")
        print(f"   [INFO] Speakers: {conversation.get('speaker_a', 'Unknown')} & {conversation.get('speaker_b', 'Unknown')}")

        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conversation[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conversation:
                session_time = parse_locomo_timestamp(conversation[session_time_key])

                for i, msg in enumerate(session_messages):
                    msg_timestamp = session_time + timedelta(seconds=i * 30)
                    iso_timestamp = to_iso_format(msg_timestamp)

                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        speaker_name_to_id[speaker_name] = f"{speaker_name.lower().replace(' ', '_')}_{con_id}"

                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "speaker_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conversation[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]
                    messages.append(message)

        raw_data_dict[str(con_id)] = messages
        print(f"   [SUCCESS] Converted {len(messages)} messages from {len(session_keys)} sessions")

    return raw_data_dict


def convert_conversation_to_raw_data_list(conversation: list) -> List[RawData]:
    return [RawData(content=msg, data_id=str(uuid.uuid4())) for msg in conversation]


# ==================== Episode Extraction ====================

async def episode_extraction_from_conversation(
    raw_data_list: List[RawData],
    llm_provider: LLMProvider = None,
    episode_extractor: ConvEpisodeExtractor = None,
    smart_mask: bool = True,
    conv_id: str = None,
    progress: Progress = None,
    task_id: int = None,
) -> list:
    """
    Run boundary detection on a conversation and extract Episodes.
    """
    if episode_extractor is None:
        episode_extractor = ConvEpisodeExtractor(llm_provider=llm_provider)

    episode_list = []
    speakers = {
        raw_data.content["speaker_id"]
        for raw_data in raw_data_list
        if isinstance(raw_data.content, dict) and "speaker_id" in raw_data.content
    }
    history_raw_data_list = []

    total_messages = len(raw_data_list)
    smart_mask_flag = False

    for idx, raw_data in enumerate(raw_data_list):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)

        if history_raw_data_list == [] or len(history_raw_data_list) == 1:
            history_raw_data_list.append(raw_data)
            continue

        if smart_mask and len(history_raw_data_list) > 5:
            smart_mask_flag = True
        else:
            smart_mask_flag = False

        request = ConvEpisodeExtractRequest(
            history_raw_data_list=history_raw_data_list,
            new_raw_data_list=[raw_data],
            user_id_list=list(speakers),
            smart_mask_flag=smart_mask_flag,
        )

        for i in range(5):
            try:
                result = await episode_extractor.extract_episode(request)
                break
            except Exception as e:
                console.print(f"  [yellow][!] Conv-{conv_id} msg {idx}: retry {i+1}/5: {e}[/yellow]")
                if i == 4:
                    raise RuntimeError("Episode extraction failed after 5 retries")
                continue

        episode_result = result[0]

        if episode_result is None:
            history_raw_data_list.append(raw_data)
        elif isinstance(episode_result, Episode):
            if smart_mask_flag:
                history_raw_data_list = [history_raw_data_list[-1], raw_data]
            else:
                history_raw_data_list = [raw_data]
            # summary is already set by _generate_episode_memory as content[:200]+"..."
            episode_list.append(episode_result)
        else:
            console.print(f"  [red][ERROR] Unexpected result type: {episode_result}[/red]")
            raise RuntimeError("Episode extraction returned unexpected result")

    # Update progress to 100%
    if progress and task_id is not None:
        progress.update(task_id, completed=total_messages)

    # Handle remaining messages as the final episode
    if history_raw_data_list:
        episode = Episode(
            type=RawDataType.CONVERSATION,
            event_id=str(uuid.uuid4()),
            user_id_list=list(speakers),
            original_data=history_raw_data_list,
            timestamp=(episode_list[-1].timestamp) if episode_list else datetime.now(),
            summary="(pending)",
        )

        # Generate episode memory for the final segment
        try:
            processed_data = [episode_extractor._data_process(rd) for rd in history_raw_data_list]
            processed_data = [d for d in processed_data if d is not None]
            episode = await episode_extractor._generate_episode_memory(episode, processed_data)
            episode.original_data = processed_data
        except Exception as e:
            console.print(f"  [yellow][!] Final segment episode memory generation failed: {e}[/yellow]")
            episode.original_data = [episode_extractor._data_process(rd) for rd in history_raw_data_list]
            episode.original_data = [d for d in episode.original_data if d is not None]

        episode_list.append(episode)

    return episode_list


# ==================== Single Conversation Processing ====================

async def process_single_conversation(
    conv_id: str,
    conversation: list,
    save_dir: Path,
    llm_provider: LLMProvider = None,
    progress_counter: dict = None,
    progress: Progress = None,
    task_id: int = None,
) -> tuple:
    """Process a single conversation and return results."""
    try:
        if progress and task_id is not None:
            progress.update(task_id, status="Processing")

        raw_data_list = convert_conversation_to_raw_data_list(conversation)
        episode_extractor = ConvEpisodeExtractor(llm_provider=llm_provider)
        episode_list = await episode_extraction_from_conversation(
            raw_data_list,
            llm_provider=llm_provider,
            episode_extractor=episode_extractor,
            conv_id=conv_id,
            progress=progress,
            task_id=task_id,
        )

        # Normalize timestamps before saving
        for ep in episode_list:
            if hasattr(ep, 'timestamp'):
                ts = ep.timestamp
                if isinstance(ts, (int, float)):
                    ep.timestamp = datetime.fromtimestamp(ts)
                elif isinstance(ts, str):
                    ep.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    ep.timestamp = get_now_with_timezone()

        # Save
        episode_dicts = [ep.to_dict() for ep in episode_list]
        output_file = save_dir / f"episode_list_conv_{conv_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(episode_dicts, f, ensure_ascii=False, indent=2)

        if progress_counter:
            progress_counter['completed'] += 1

        return conv_id, episode_list

    except Exception as e:
        console.print(f"\n[red][ERROR] Conversation {conv_id} failed: {e}[/red]")
        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1
        import traceback
        traceback.print_exc()
        return conv_id, []


# ==================== Main ====================

async def main():
    config = ExperimentConfig()
    llm_service = config.llm_service
    dataset_path = config.dataset_path
    raw_data_dict = load_locomo_raw_data(dataset_path)

    save_dir = config.episodes_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 1: Memory Extraction (Episode Boundary Detection)[/bold cyan]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]\n")

    console.print(f"[INFO] Total conversations: {len(raw_data_dict)}", style="bold cyan")
    total_messages = sum(len(conv) for conv in raw_data_dict.values())
    console.print(f"[INFO] Total messages: {total_messages}", style="bold blue")
    console.print(f"[INFO] Save directory: {save_dir}", style="bold green")

    # Initialize shared LLM Provider
    console.print("[INFO] Initializing LLM Provider...", style="yellow")
    console.print(f"   Model: {config.llm_config[llm_service]['model']}", style="dim")
    console.print(f"   Base URL: {config.llm_config[llm_service]['base_url']}", style="dim")

    shared_llm_provider = LLMProvider(
        provider_type="openai",
        model=config.llm_config[llm_service]["model"],
        api_key=config.llm_config[llm_service]["api_key"],
        base_url=config.llm_config[llm_service]["base_url"],
        temperature=config.llm_config[llm_service]["temperature"],
        max_tokens=config.llm_config[llm_service]["max_tokens"],
    )

    progress_counter = {
        'total': len(raw_data_dict),
        'completed': 0,
        'failed': 0
    }

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TaskProgressColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        TextColumn("*"),
        TimeRemainingColumn(),
        TextColumn("*"),
        TextColumn("[bold blue]{task.fields[status]}"),
        console=console,
        transient=False,
        refresh_per_second=1,
    ) as progress:
        main_task = progress.add_task(
            "[bold cyan][MAIN] Total Progress",
            total=len(raw_data_dict),
            completed=0,
            status="Processing",
        )

        conversation_tasks = {}
        updated_tasks = []

        for conv_id, conversation in raw_data_dict.items():
            conv_task_id = progress.add_task(
                f"[yellow]Conv-{conv_id}",
                total=len(conversation),
                completed=0,
                status="Waiting",
            )
            conversation_tasks[conv_id] = conv_task_id

            task = process_single_conversation(
                conv_id,
                conversation,
                save_dir,
                llm_provider=shared_llm_provider,
                progress_counter=progress_counter,
                progress=progress,
                task_id=conv_task_id,
            )
            updated_tasks.append(task)

        async def run_with_completion(task, conv_id):
            result = await task
            progress.update(conversation_tasks[conv_id],
                            status="[green]Done[/green]",
                            completed=progress.tasks[conversation_tasks[conv_id]].total)
            progress.update(main_task, advance=1)
            return result

        results = await asyncio.gather(*[
            run_with_completion(task, conv_id)
            for (conv_id, _), task in zip(raw_data_dict.items(), updated_tasks)
        ])

        progress.update(main_task, status="[green]Completed[/green]")

    elapsed = time.time() - start_time

    # Summary
    all_episodes = []
    successful = 0
    for conv_id, ep_list in results:
        if ep_list:
            successful += 1
            all_episodes.extend(ep_list)

    console.print("\n" + "=" * 60, style="dim")
    console.print("[STATS] Processing Statistics:", style="bold")
    console.print(f"   [SUCCESS] Conversations: {successful}/{len(raw_data_dict)}", style="green")
    console.print(f"   [INFO] Total episodes: {len(all_episodes)}", style="blue")
    console.print(f"   [TIME] Elapsed: {elapsed:.2f}s", style="yellow")
    console.print(f"   [TIME] Average per conversation: {elapsed / len(raw_data_dict):.2f}s", style="cyan")
    console.print("=" * 60, style="dim")

    # Save summary
    all_dicts = [ep.to_dict() for ep in all_episodes]
    summary_file = save_dir / "episode_list_all.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_dicts, f, ensure_ascii=False, indent=2)
    console.print(f"\n[SAVE] Summary saved to: {summary_file}", style="green")

    # Save processing summary
    summary = {
        "total_conversations": len(raw_data_dict),
        "successful_conversations": successful,
        "total_episodes": len(all_episodes),
        "processing_time_seconds": elapsed,
        "average_time_per_conversation": elapsed / len(raw_data_dict),
        "conversation_results": {
            conv_id: len(ep_list) for conv_id, ep_list in results
        }
    }
    summary_info_file = save_dir / "processing_summary.json"
    with open(summary_info_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[SAVE] Processing summary saved to: {summary_info_file}\n", style="green")


if __name__ == "__main__":
    asyncio.run(main())
