# src/file_discovery.py
import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

class FileDiscovery:
    """Discover molecule data files based on patterns"""

    def __init__(self, data_dir: str, molecule_mapping: Dict):
        self.data_dir = Path(data_dir)
        self.molecule_mapping = molecule_mapping

    
    
    def discover_molecule_files(self, molecule: str) -> List[str]:
        """
        Discover all files for a given molecule
        
        Args:
            molecule: Molecule name (e.g., 'azithromycin')
        
        Returns:
            List of file paths matching the molecule patterns
        """
        if molecule not in self.molecule_mapping['molecules']:
            return []
        
        patterns = self.molecule_mapping['molecules'][molecule]['file_patterns']
        found_files = []
        
        for pattern in patterns:
            file_path = os.path.join(self.data_dir, pattern)
            found_files.extend(glob.glob(file_path))
        
        return sorted(found_files)
    
    def discover_cipla_file(self) -> str:
        """Discover Cipla GRN file"""
        cipla_patterns = ["cipla_api_grn*.xlsx", "cipla_grn*.xlsx"]
        
        for pattern in cipla_patterns:
            file_path = os.path.join(self.data_dir, pattern)
            files = glob.glob(file_path)
            if files:
                return files[0]
        
        return None
    
    def get_available_molecules(self) -> Dict[str, Dict]:
        """Get all available molecules with their file count"""
        available = {}
        
        for mol_name, mol_config in self.molecule_mapping.items():
            pattern = mol_config.get("file_pattern", f"*{mol_name}*")
            files = list(self.data_dir.rglob(pattern))

            if files:
                available[mol_name] = {
                    "description": mol_config.get("description", ""),
                    "file_count": len(files),
                    "cipla_available": "cipla_api_filter" in mol_config
                }

        return available

    
    def get_molecule_file_info(self, molecule: str) -> Dict:
        """Get detailed file information for a molecule"""
        files = self.discover_molecule_files(molecule)
        
        info = {
            'molecule': molecule,
            'total_files': len(files),
            'files': [],
            'cipla_available': self.discover_cipla_file() is not None,
            'size_bytes': 0,
            'last_modified': None
        }
        
        for file in files:
            file_stat = os.stat(file)
            info['files'].append({
                'name': os.path.basename(file),
                'path': file,
                'size_bytes': file_stat.st_size,
                'modified': file_stat.st_mtime
            })
            info['size_bytes'] += file_stat.st_size
        
        return info