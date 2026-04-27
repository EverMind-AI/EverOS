"""
Result saver utilities - unified result saving with JSON, pickle support.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict


class ResultSaver:
    """Result saver."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize saver.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Any, filename: str):
        """
        Save JSON file.
        
        Args:
            data: Data to save
            filename: Filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def load_json(self, filename: str) -> Any:
        """
        Load JSON file.
        
        Args:
            filename: Filename
            
        Returns:
            Loaded data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_pickle(self, data: Any, filename: str):
        """
        Save pickle file.
        
        Args:
            data: Data to save
            filename: Filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_pickle(self, filename: str) -> Any:
        """
        Load pickle file.
        
        Args:
            filename: Filename
            
        Returns:
            Loaded data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""
        return (self.output_dir / filename).exists()

