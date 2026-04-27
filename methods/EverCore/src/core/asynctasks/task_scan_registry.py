from typing import List


class TaskScanDirectoriesRegistry:
    """Scan directory registry"""

    def __init__(self):
        """Initialize scan directory registry"""
        self.scan_directories: List[str] = []

    def add_scan_path(self, path: str) -> 'TaskScanDirectoriesRegistry':
        """Add scan directory"""
        self.scan_directories.append(path)
        return self

    def get_scan_directories(self) -> List[str]:
        """Get scan directories"""
        return self.scan_directories

    def clear(self) -> 'TaskScanDirectoriesRegistry':
        """Clear scan directories"""
        self.scan_directories = []
        return self
