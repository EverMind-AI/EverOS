"""Terminal UI Tools

Provides beautiful terminal output formatting.
"""

import re
from typing import List, Dict, Any, Optional

from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI


def extract_event_time_from_memory(mem: Dict[str, Any]) -> Optional[str]:
    """Extract actual event time from memory data
    
    Extraction priority:
    1. Date in 'subject' field (parentheses format, e.g., "(2025-08-26)")
    2. Date in 'subject' field (Chinese format, e.g., "2025年8月26日")
    3. Date in 'episode' content (Chinese or ISO format)
    4. Time fields: timestamp / created_at / event_time
    5. Return None if extraction fails
    
    Args:
        mem: Memory dictionary containing subject, episode, timestamp, etc.
        
    Returns:
        Date string in YYYY-MM-DD format, or None (if extraction fails)
        
    Examples:
        >>> mem = {"subject": "Beijing Travel Advice (2025-08-26)"}
        >>> extract_event_time_from_memory(mem)
        '2025-08-26'
        
        >>> mem = {"timestamp": "2025-08-26T10:30:00"}
        >>> extract_event_time_from_memory(mem)
        '2025-08-26'
        
        >>> mem = {"subject": "", "episode": ""}
        >>> extract_event_time_from_memory(mem)
        None
    """
    subject = mem.get("subject", "")
    episode = mem.get("episode", "")
    
    # 1. Extract from subject: Match ISO date format inside parentheses (YYYY-MM-DD)
    if subject:
        match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', subject)
        if match:
            return match.group(1)
        
        # 2. Extract from subject: Match Chinese date format "YYYY年MM月DD日"
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', subject)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 3. Extract from episode (search entire content, no character limit)
    if episode:
        # Match "于YYYY年MM月DD日" or "在YYYY年MM月DD日" (At YYYY...)
        match = re.search(r'[于在](\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Match ISO format "YYYY-MM-DD"
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', episode)
        if match:
            return match.group(0)
        
        # Match other Chinese date formats (without "at" prefix)
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 4. Extract from time fields (timestamp, created_at, event_time, updated_at)
    for time_field in ["timestamp", "event_time", "created_at", "updated_at"]:
        time_value = mem.get(time_field, "")
        if time_value:
            # Support ISO format "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
            match = re.search(r'(\d{4}-\d{2}-\d{2})', str(time_value))
            if match:
                return match.group(1)
    
    # 5. Failed to extract event time, return None
    return None


class ChatUI:
    """Terminal Interface Utility Class"""
    
    @staticmethod
    def _ui() -> CLIUI:
        """Get UI instance"""
        return CLIUI()
    
    @staticmethod
    def clear_screen():
        """Clear screen"""
        print("\033[2J\033[H", end="")
        import sys
        sys.stdout.flush()
    
    @staticmethod
    def print_banner(texts: I18nTexts):
        """Print welcome banner"""
        ui = ChatUI._ui()
        print()
        ui.banner(texts.get("banner_title"), subtitle=texts.get("banner_subtitle"))
        print()
    
    @staticmethod
    def print_group_list(groups: List[Dict[str, Any]], texts: I18nTexts):
        """Display group list"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("groups_available_title"))
        
        rows = []
        for group in groups:
            index = group["index"]
            group_id = group["group_id"]
            name = group.get("name", group_id)
            count = group["memcell_count"]
            count_text = f"💾 {count} " + ("memories" if texts.language == "en" else "条记忆")
            rows.append([f"[{index}]", group_id, f'📝 "{name}"', count_text])
        
        headers = [
            texts.get("table_header_index"),
            texts.get("table_header_group"),
            texts.get("table_header_name"),
            texts.get("table_header_count"),
        ]
        ui.table(headers=headers, rows=rows, aligns=["right", "left", "left", "right"])
    
    @staticmethod
    def print_retrieved_memories(
        memories: List[Dict[str, Any]],
        texts: I18nTexts,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Display retrieved memories"""
        ui = ChatUI._ui()
        
        heading = f"🔍 {texts.get('retrieval_complete')}"
        shown_count = len(memories)
        if shown_count > 0:
            heading += f" - {texts.get('retrieval_showing', shown=shown_count)}"
        
        # Display retrieval mode and latency
        if retrieval_metadata:
            retrieval_mode = retrieval_metadata.get("retrieval_mode", "rrf")
            latency_ms = retrieval_metadata.get("total_latency_ms", 0.0)
            
            # Internationalized retrieval mode display
            mode_map = {
                "keyword": texts.get("agentic_mode_keyword"),
                "vector": texts.get("agentic_mode_vector"),
                "hybrid": texts.get("agentic_mode_hybrid"),
                "rrf": texts.get("agentic_mode_rrf"),
                "agentic": texts.get("agentic_mode_agentic"),
            }
            mode_text = mode_map.get(retrieval_mode, retrieval_mode)
            heading += f" | {mode_text} | {int(latency_ms)}ms"
        
        ui.section_heading(heading)
        
        # 🔥 Agentic Retrieval Special Info Display
        if retrieval_metadata and retrieval_metadata.get("retrieval_mode") == "agentic":
            agentic_info = []
            
            # LLM Judgment Result (Internationalized)
            is_sufficient = retrieval_metadata.get("is_sufficient")
            if is_sufficient is not None:
                status_icon = "✅" if is_sufficient else "❌"
                status_text = texts.get("agentic_sufficient") if is_sufficient else texts.get("agentic_insufficient")
                agentic_info.append(f"{texts.get('agentic_llm_judgment')}: {status_icon} {status_text}")
            
            # Multi-round Check (Internationalized)
            is_multi_round = retrieval_metadata.get("is_multi_round", False)
            if is_multi_round:
                agentic_info.append(f"🔄 {texts.get('agentic_multi_round')}")
                
                # Refined Queries
                refined_queries = retrieval_metadata.get("refined_queries", [])
                if refined_queries:
                    agentic_info.append(f"{texts.get('agentic_generated_queries')}: {len(refined_queries)}")
            else:
                agentic_info.append(f"⚡ {texts.get('agentic_single_round')}")
            
            # Round Statistics (Internationalized)
            round1_count = retrieval_metadata.get("round1_count", 0)
            round2_count = retrieval_metadata.get("round2_count", 0)
            items_text = texts.get("agentic_items")
            if round1_count:
                agentic_info.append(f"{texts.get('agentic_round1_count')}: {round1_count} {items_text}")
            if round2_count:
                agentic_info.append(f"{texts.get('agentic_round2_count')}: {round2_count} {items_text}")
            
            if agentic_info:
                print()
                ui.note(" | ".join(agentic_info), icon="🤖")
                
                # Display LLM Reasoning (Internationalized optimization hint)
                reasoning = retrieval_metadata.get("reasoning")
                if reasoning:
                    # Optimize common misleading hints (Internationalized)
                    # Detect Chinese content and replace with internationalized text
                    chinese_keywords = [
                        "为空", "均为空", "内容为空", "记忆内容",
                        "未提供", "不足", "无法提供", "相关性",
                        "检索到的记忆", "信息不够"
                    ]
                    if any(kw in reasoning for kw in chinese_keywords):
                        reasoning = texts.get("agentic_reasoning_hint")
                    
                    print(f"   💭 {reasoning}")
                
                # Display Refined Queries (Internationalized)
                if is_multi_round:
                    refined_queries = retrieval_metadata.get("refined_queries", [])
                    if refined_queries:
                        print(f"   🔍 {texts.get('agentic_supplementary_queries')} ({len(refined_queries)}):")
                        for i, q in enumerate(refined_queries[:3], 1):
                            print(f"      {i}. {q[:60]}{'...' if len(q) > 60 else ''}")
        
        # Display Memory List
        lines = []
        for i, mem in enumerate(memories, start=1):
            # Extract actual event time (not storage time)
            event_time = extract_event_time_from_memory(mem)
            
            # Priority: subject > summary > episode > atomic_fact > content
            # Use strip() to ensure empty strings are handled correctly
            subject = (mem.get("subject") or "").strip()
            summary = (mem.get("summary") or "").strip()
            episode = (mem.get("episode") or "").strip()
            foresight = (mem.get("foresight") or "").strip()  # Foresight content field
            atomic_fact = (mem.get("atomic_fact") or "").strip()
            content = (mem.get("content") or "").strip()
            
            # Select first non-empty field
            display_text = subject or summary or episode or foresight or atomic_fact or content or "(No Content)"
            
            # Limit display length
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."
            
            # Build display line: show time if available, otherwise omit
            score = mem.get("score")
            score_text = ""
            if isinstance(score, (int, float)):
                score_text = f"{score:.3f} | "
            if event_time:
                lines.append(f"📌 [{i}]  {event_time}  │  {score_text}{display_text}")
            else:
                lines.append(f"📌 [{i}]  {score_text}{display_text}")
        
        if lines:
            print()
            ui.panel(lines)
    
    @staticmethod
    def print_generating_indicator(texts: I18nTexts):
        """Display generation progress indicator"""
        ui = ChatUI._ui()
        print()
        ui.note(f"🤔 {texts.get('chat_generating')}", icon="⏳")
    
    @staticmethod
    def print_generation_complete(texts: I18nTexts):
        """Clear generation indicator and show completion mark"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
        ui = ChatUI._ui()
        ui.success(f"✓ {texts.get('chat_generation_complete')}")
    
    @staticmethod
    def clear_progress_indicator():
        """Clear progress indicator"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
    
    @staticmethod
    def print_assistant_response(response: str, texts: I18nTexts):
        """Display Assistant Response
        
        Optimized display:
        - Mainly show 'answer' (Large Title)
        - 'references' and 'confidence' as metadata (Small text)
        - Hide 'reasoning'
        """
        ui = ChatUI._ui()
        print()
        
        # Try parsing JSON response
        try:
            import json
            data = json.loads(response)
            
            # Extract fields
            answer = data.get("answer", "")
            references = data.get("references", [])
            confidence = data.get("confidence", "")
            
            # Display main answer (Large Title)
            ui.panel([answer], title=f"🤖 {texts.get('response_assistant_title')}")
            
            # Display metadata (Small text, dimmed)
            metadata_parts = []
            if references:
                ref_text = ", ".join(references)
                metadata_parts.append(f"📚 {ref_text}")
            if confidence:
                confidence_icon = {"high": "✓", "medium": "~", "low": "?"}.get(confidence, "")
                metadata_parts.append(f"{confidence_icon} {confidence}")
            
            if metadata_parts:
                metadata_line = "  │  ".join(metadata_parts)
                print(f"  {metadata_line}")
            
        except (json.JSONDecodeError, ValueError):
            # If not JSON format, display raw response directly
            ui.panel([response], title=f"🤖 {texts.get('response_assistant_title')}")
        
        ui.rule()
        print()
    
    @staticmethod
    def print_help(texts: I18nTexts):
        """Display help information"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("cmd_help_title"))
        lines = [
            f"🚪  {texts.get('cmd_exit')}",
            f"🧹  {texts.get('cmd_clear')}",
            f"🔄  {texts.get('cmd_reload')}",
            f"❓  {texts.get('cmd_help')}",
        ]
        ui.panel(lines)
        print()
    
    @staticmethod
    def print_info(message: str, texts: I18nTexts):
        """Display informational message"""
        ui = ChatUI._ui()
        print()
        ui.success(f"✓ {message}")
        print()
    
    @staticmethod
    def print_error(message: str, texts: I18nTexts):
        """Display error message"""
        ui = ChatUI._ui()
        print()
        ui.error(f"✗ {message}")
        print()
