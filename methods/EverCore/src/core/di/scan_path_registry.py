from typing import List


class ScannerPathsRegistry:
    """Scanner path registry"""

    def __init__(self):
        """Initialize scanner path registry"""
        self.scan_paths: List[str] = []

    def add_scan_path(self, path: str) -> 'ScannerPathsRegistry':
        """Add scan path"""
        self.scan_paths.append(path)
        return self

    def get_scan_paths(self) -> List[str]:
        """Get scan paths"""
        return self.scan_paths

    def clear(self) -> 'ScannerPathsRegistry':
        """Clear scan paths"""
        self.scan_paths = []
        return self
