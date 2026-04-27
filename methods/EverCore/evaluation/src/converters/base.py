"""
Base Converter - define dataset converter base class interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseConverter(ABC):
    """Dataset converter base class."""
    
    @abstractmethod
    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        Convert dataset to Locomo format.
        
        Args:
            input_paths: Input file path dict, e.g., {"raw": "path/to/raw.json"}
            output_path: Output file path
        """
        pass
    
    @abstractmethod
    def get_input_files(self) -> Dict[str, str]:
        """
        Return required input file list.
        
        Returns:
            File name mapping, e.g., {"raw": "longmemeval_s_cleaned.json"}
        """
        pass
    
    def get_output_filename(self) -> str:
        """
        Return output filename (converted version).
        
        Returns:
            Filename, e.g., "longmemeval_s_locomo_style.json"
        """
        return "converted_locomo_style.json"
    
    def needs_conversion(self, data_dir: Path) -> bool:
        """
        Check if conversion needed (whether converted file exists).
        
        Args:
            data_dir: Dataset directory
            
        Returns:
            True if conversion needed, False if converted version exists
        """
        output_file = data_dir / self.get_output_filename()
        return not output_file.exists()
    
    def get_converted_path(self, data_dir: Path) -> Path:
        """
        Get converted file path.
        
        Args:
            data_dir: Dataset directory
            
        Returns:
            Converted file path
        """
        return data_dir / self.get_output_filename()


