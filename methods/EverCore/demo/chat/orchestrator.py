"""Chat Application Orchestrator

Responsible for the orchestration of the entire chat application:
1. Initialization configuration
2. User interaction (language, scenario, group, retrieval mode selection)
3. Session management
4. Conversation loop
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from demo.config import ChatModeConfig, LLMConfig, MongoDBConfig
from demo.utils import ensure_mongo_beanie_ready
from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI

from .session import ChatSession
from .ui import ChatUI
from .selectors import LanguageSelector, ScenarioSelector, GroupSelector


class ChatOrchestrator:
    """Chat Application Orchestrator"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "demo" / ".chat_history"
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging - Hide DEBUG logs from third-party libraries"""
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        logging.getLogger().setLevel(logging.WARNING)
        
        # Disable common third-party library logs
        for logger_name in ['jieba', 'elasticsearch', 'urllib3', 'pymongo', 'pymilvus']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    def setup_readline(self):
        """Configure readline history"""
        try:
            import readline
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            readline.set_history_length(1000)
        except Exception:
            pass
    
    def save_readline_history(self):
        """Save readline history"""
        try:
            import readline
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass
    
    async def select_language(self) -> I18nTexts:
        """Language selection"""
        language = LanguageSelector.select_language()
        return I18nTexts(language)
    
    async def select_scenario(self, texts: I18nTexts) -> Optional[str]:
        """Scenario selection"""
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        scenario_type = ScenarioSelector.select_scenario(texts)
        if not scenario_type:
            ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
            return None
        
        return scenario_type
    
    async def initialize_database(self, texts: I18nTexts) -> bool:
        """Initialize database connection"""
        mongo_config = MongoDBConfig()
        
        try:
            await ensure_mongo_beanie_ready(mongo_config)
            return True
        except Exception as e:
            ChatUI.print_error(texts.get("mongodb_init_failed", error=str(e)), texts)
            return False
    
    async def select_group(self, texts: I18nTexts) -> Optional[str]:
        """Group selection"""
        groups = await GroupSelector.list_available_groups()
        selected_group_id = await GroupSelector.select_group(groups, texts)
        
        if not selected_group_id:
            ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
            return None
        
        return selected_group_id
    
    async def select_retrieval_mode(self, texts: I18nTexts) -> str:
        """Retrieval mode selection
        
        Args:
            texts: I18nTexts object
            
        Returns:
            Retrieval mode string
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("retrieval_mode_selection_title"))
        print()
        print(f"  [1] {texts.get('retrieval_mode_keyword')} - {texts.get('retrieval_mode_keyword_desc')}")
        print(f"  [2] {texts.get('retrieval_mode_vector')} - {texts.get('retrieval_mode_vector_desc')}")
        print(f"  [3] {texts.get('retrieval_mode_hybrid')} - {texts.get('retrieval_mode_hybrid_desc')}")
        print(f"  [4] {texts.get('retrieval_mode_rrf')} - {texts.get('retrieval_mode_rrf_desc')}")
        print(f"  [5] {texts.get('retrieval_mode_agentic')} - {texts.get('retrieval_mode_agentic_desc')}")
        print()
        
        mode_map = {1: "keyword", 2: "vector", 3: "hybrid", 4: "rrf", 5: "agentic"}
        mode_desc = {
            1: texts.get('retrieval_mode_keyword'),
            2: texts.get('retrieval_mode_vector'),
            3: texts.get('retrieval_mode_hybrid'),
            4: texts.get('retrieval_mode_rrf'),
            5: texts.get('retrieval_mode_agentic'),
        }
        
        while True:
            try:
                choice = input(f"{texts.get('retrieval_mode_prompt')}: ").strip()
                if not choice:
                    continue
                
                index = int(choice)
                if index in mode_map:
                    # Special hint: Agentic mode requires LLM
                    if index == 5:
                        print()
                        ui.note(texts.get("retrieval_mode_agentic_cost_warning"), icon="💰")
                        print()
                    
                    ui.success(f"✓ {texts.get('retrieval_mode_selected')}: {mode_desc[index]}")
                    return mode_map[index]
                else:
                    ui.error(f"✗ {texts.get('retrieval_mode_invalid_range')}")
            except ValueError:
                ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                raise
    
    def verify_api_key(self, llm_config: LLMConfig, texts: I18nTexts) -> bool:
        """Verify if API Key is configured"""
        import os
        api_key_present = any([
            llm_config.api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ])
        
        if not api_key_present:
            ChatUI.print_error(texts.get("config_api_key_missing"), texts)
            print(f"{texts.get('config_api_key_hint')}\n")
            return False
        
        return True
    
    async def create_session(
        self,
        group_id: str,
        scenario_type: str,
        retrieval_mode: str,
        texts: I18nTexts,
    ) -> Optional[ChatSession]:
        """Create and initialize session"""
        chat_config = ChatModeConfig()
        llm_config = LLMConfig()
        
        # Extract user_id from group_id (format: chat_user_001_assistant -> user_001)
        user_id = "user_001"  # Default
        if "_" in group_id:
            parts = group_id.split("_")
            # Try to find user_XXX pattern
            for i, part in enumerate(parts):
                if part == "user" and i + 1 < len(parts):
                    user_id = f"user_{parts[i + 1]}"
                    break
        
        session = ChatSession(
            group_id=group_id,
            config=chat_config,
            llm_config=llm_config,
            scenario_type=scenario_type,
            retrieval_mode=retrieval_mode,
            data_source="episodic_memory",  # Use API memory_types value
            texts=texts,
            user_id=user_id,
        )
        
        if not await session.initialize():
            ChatUI.print_error(texts.get("session_init_failed"), texts)
            return None
        
        return session
    
    async def run_chat_loop(self, session: ChatSession, texts: I18nTexts):
        """Run conversation loop"""
        # Clear screen, enter clean chat interface
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        # Show start note
        ui = CLIUI()
        print()
        ui.rule()
        ui.note(texts.get("chat_start_note"), icon="💬")
        ui.rule()
        print()
        
        while True:
            try:
                user_input = input(texts.get("chat_input_prompt")).strip()
                
                if not user_input:
                    continue
                
                command = user_input.lower()
                
                # Handle commands
                if command == "exit":
                    await self._handle_exit(session, texts)
                    break
                elif command == "clear":
                    session.clear_history()
                    continue
                elif command == "reload":
                    await session.reload_data()
                    continue
                elif command == "help":
                    ChatUI.print_help(texts)
                    continue
                
                # Execute chat
                response = await session.chat(user_input)
                ChatUI.print_assistant_response(response, texts)
            
            except KeyboardInterrupt:
                await self._handle_interrupt(session, texts)
                break
            
            except Exception as e:
                ChatUI.print_error(texts.get("chat_error", error=str(e)), texts)
                import traceback
                traceback.print_exc()
                print()
    
    async def _handle_exit(self, session: ChatSession, texts: I18nTexts):
        """Handle exit command"""
        ui = CLIUI()
        print()
        ui.note(texts.get("cmd_exit_saving"), icon="💾")
        await session.save_conversation_history()
        print()
        ui.success(f"✓ {texts.get('cmd_exit_complete')}")
        print()
    
    async def _handle_interrupt(self, session: ChatSession, texts: I18nTexts):
        """Handle interrupt signal"""
        ui = CLIUI()
        print("\n")
        ui.note(texts.get("cmd_interrupt_saving"), icon="⚠️")
        await session.save_conversation_history()
        print()
        ui.success(f"✓ {texts.get('cmd_exit_complete')}")
        print()
    
    async def run(self):
        """Run chat application main flow"""
        # 1. Initialize readline
        self.setup_readline()
        
        # 2. Clear screen, then language selection
        ChatUI.clear_screen()
        texts = await self.select_language()
        
        # 3. Scenario selection
        scenario_type = await self.select_scenario(texts)
        if not scenario_type:
            return
        
        # 4. Clear screen
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        # 5. Verify API Key
        llm_config = LLMConfig()
        if not self.verify_api_key(llm_config, texts):
            return
        
        # 6. Initialize database
        if not await self.initialize_database(texts):
            return
        
        # 7. Group selection
        group_id = await self.select_group(texts)
        if not group_id:
            return
        
        # 8. Retrieval mode selection
        try:
            retrieval_mode = await self.select_retrieval_mode(texts)
        except KeyboardInterrupt:
            print("\n")
            return
        
        # 9. Create session
        session = await self.create_session(group_id, scenario_type, retrieval_mode, texts)
        if not session:
            return
        
        # 10. Run conversation loop
        await self.run_chat_loop(session, texts)
        
        # 11. Save history
        self.save_readline_history()

