"""
Prompt utilities - provide prompt loading and formatting.
"""
from pathlib import Path
from typing import Dict, Any
import yaml


class PromptManager:
    """Prompt manager."""
    
    _instance = None
    _prompts = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts config file."""
        # Find config/prompts.yaml
        current_file = Path(__file__)
        config_path = current_file.parent.parent.parent / "config" / "prompts.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Prompts config not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._prompts = yaml.safe_load(f)
    
    def get_prompt(self, prompt_key: str, sub_key: str = None) -> str:
        """
        Get prompt template.
        
        Args:
            prompt_key: Prompt category key (e.g., "answer_generation", "llm_judge")
            sub_key: Sub-key (e.g., "system_prompt", "user_prompt")
            
        Returns:
            Prompt template string
        
        Example:
            >>> pm = PromptManager()
            >>> pm.get_prompt("answer_generation", "template")
            'Based on the following memories...'
            >>> pm.get_prompt("llm_judge", "system_prompt")
            'You are an expert grader...'
        """
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in prompts.yaml")
        
        prompt_config = self._prompts[prompt_key]
        
        if sub_key:
            if sub_key not in prompt_config:
                raise KeyError(
                    f"Sub-key '{sub_key}' not found in prompt '{prompt_key}'"
                )
            return prompt_config[sub_key].strip()
        
        # If no sub_key, default to 'template'
        if "template" in prompt_config:
            return prompt_config["template"].strip()
        
        raise KeyError(
            f"No 'template' field found in prompt '{prompt_key}' "
            f"and no sub_key specified"
        )
    
    def format_prompt(
        self, 
        prompt_key: str, 
        sub_key: str = None, 
        **kwargs
    ) -> str:
        """
        Get and format prompt.
        
        Args:
            prompt_key: Prompt category key
            sub_key: Sub-key
            **kwargs: Formatting parameters
            
        Returns:
            Formatted prompt
        
        Example:
            >>> pm = PromptManager()
            >>> pm.format_prompt(
            ...     "answer_generation",
            ...     context="Memory 1...",
            ...     question="What is X?"
            ... )
            'Based on the following memories...Memory 1...Question: What is X?'
        """
        template = self.get_prompt(prompt_key, sub_key)
        return template.format(**kwargs)


# Global instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(prompt_key: str, sub_key: str = None) -> str:
    """Shortcut: get prompt."""
    return get_prompt_manager().get_prompt(prompt_key, sub_key)


def format_prompt(prompt_key: str, sub_key: str = None, **kwargs) -> str:
    """Shortcut: format prompt."""
    return get_prompt_manager().format_prompt(prompt_key, sub_key, **kwargs)


