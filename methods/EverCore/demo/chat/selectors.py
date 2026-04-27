"""Interactive Selectors

Provides selection for language, scenario, and groups.
"""

from typing import List, Dict, Any, Optional

from demo.config import ScenarioType
from demo.utils import query_all_groups_from_mongodb
from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI


class LanguageSelector:
    """Language Selector"""

    @staticmethod
    def select_language() -> str:
        """Interactive language selection

        Returns:
            Language code: "zh" or "en"
        """
        print()
        print("=" * 60)
        print("  🌏  语言选择 / Language Selection")
        print("=" * 60)
        print()
        print("  [1] 中文 (Chinese)")
        print("  [2] English")
        print()
        # Language consistency hint
        print("  💡 提示：为获得最佳体验，建议记忆数据与选择的语言保持一致")
        print(
            "     Note: For best experience, memory data should match the selected language"
        )
        print()

        while True:
            try:
                choice = input("请选择语言 / Please select language [1-2]: ").strip()
                if not choice:
                    continue

                index = int(choice)
                if index == 1:
                    print("\n✓ Selected: Chinese | AI will respond in Chinese\n")
                    return "zh"
                elif index == 2:
                    print("\n✓ Selected: English | AI will respond in English\n")
                    return "en"
                else:
                    print("❌ 请输入 1 或 2 / Please enter 1 or 2\n")

            except ValueError:
                print("❌ 请输入有效的数字 / Please enter a valid number\n")
            except KeyboardInterrupt:
                print("\n")
                return "zh"


class ScenarioSelector:
    """Scenario Mode Selector"""

    @staticmethod
    def select_scenario(texts: I18nTexts) -> Optional[ScenarioType]:
        """Interactive scenario selection

        Args:
            texts: I18nTexts object

        Returns:
            ScenarioType or None (Cancelled)
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("scenario_selection_title"))
        print()

        print(f"  [1] {texts.get('scenario_solo')}")
        print(f"      {texts.get('scenario_solo_desc')}")
        print()

        print(f"  [2] {texts.get('scenario_team')}")
        print(f"      {texts.get('scenario_team_desc')}")
        print()

        while True:
            try:
                choice = input(f"{texts.get('scenario_prompt')}: ").strip()
                if not choice:
                    continue

                index = int(choice)
                if index == 1:
                    ui.success(
                        f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_solo')}"
                    )
                    return ScenarioType.SOLO
                elif index == 2:
                    ui.success(
                        f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_team')}"
                    )
                    return ScenarioType.TEAM
                else:
                    ui.error(f"✗ {texts.get('invalid_input_number')}")

            except ValueError:
                ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                return None


class GroupSelector:
    """Group Selector"""

    @staticmethod
    async def list_available_groups() -> List[Dict[str, Any]]:
        """List all available groups

        Returns:
            List of groups
        """
        groups = await query_all_groups_from_mongodb()

        for idx, group in enumerate(groups, start=1):
            group["index"] = idx
            group_id = group["group_id"]
            group["name"] = "team_chat" if group_id == "AI产品群" else group_id

        return groups

    @staticmethod
    async def select_group(
        groups: List[Dict[str, Any]], texts: I18nTexts
    ) -> Optional[str]:
        """Interactive group selection

        Args:
            groups: List of groups
            texts: I18nTexts object

        Returns:
            Selected group_id or None (Cancelled)
        """
        from .ui import ChatUI

        if not groups:
            ChatUI.print_error(texts.get("groups_not_found"), texts)
            print(f"{texts.get('groups_extract_hint')}\n")
            return None

        ChatUI.print_group_list(groups, texts)

        while True:
            try:
                choice = input(
                    f"\n{texts.get('groups_select_prompt')} [1-{len(groups)}]: "
                ).strip()
                if not choice:
                    continue

                index = int(choice)
                if 1 <= index <= len(groups):
                    return groups[index - 1]["group_id"]
                else:
                    ChatUI.print_error(
                        texts.get("groups_select_range_error", min=1, max=len(groups)),
                        texts,
                    )

            except ValueError:
                ChatUI.print_error(texts.get("invalid_input_number"), texts)
            except KeyboardInterrupt:
                print("\n")
                ChatUI.print_info(texts.get("groups_selection_cancelled"), texts)
                return None
