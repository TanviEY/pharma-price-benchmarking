# src/fuzzy_matcher.py
import json
from fuzzywuzzy import fuzz
from typing import List, Tuple, Optional
from pathlib import Path

class FuzzyMatcher:
    """Fuzzy matching for molecule name inputs"""
    
    def __init__(self, threshold: int = 70):
        self.threshold = threshold
        self.molecule_mapping = self._load_molecule_mapping()
    
    def _load_molecule_mapping(self) -> dict:
        """Load molecule mapping"""
        config_path = Path("config/molecule_mapping.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def match_molecule_input(self, user_input: str) -> List[Tuple[str, int]]:
        """
        Find matching molecules based on user input
        
        Args:
            user_input: User's input string
        
        Returns:
            List of (molecule_name, confidence_score) tuples, sorted by confidence
        """
        if not user_input.strip():
            return []
        
        user_input_lower = user_input.lower().strip()
        matches = []
        
        for mol_name, mol_data in self.molecule_mapping['molecules'].items():
            # Check exact match on molecule name
            if user_input_lower == mol_name.lower():
                matches.append((mol_name, 100))
                continue
            
            # Check fuzzy match on molecule name
            name_score = fuzz.ratio(user_input_lower, mol_name.lower())
            if name_score >= self.threshold:
                matches.append((mol_name, name_score))
            
            # Check fuzzy match on aliases
            for alias in mol_data['aliases']:
                alias_score = fuzz.ratio(user_input_lower, alias.lower())
                if alias_score >= self.threshold:
                    # Use molecule name, not alias
                    if not any(m[0] == mol_name for m in matches):
                        matches.append((mol_name, alias_score))
                    break
        
        # Remove duplicates and sort by score
        unique_matches = {}
        for mol_name, score in matches:
            if mol_name not in unique_matches or score > unique_matches[mol_name]:
                unique_matches[mol_name] = score
        
        return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)
    
    def get_top_match(self, user_input: str) -> Optional[str]:
        """Get the single best match"""
        matches = self.match_molecule_input(user_input)
        return matches[0][0] if matches else None
    
    def get_aliases(self, molecule: str) -> List[str]:
        """Get all aliases for a molecule"""
        if molecule in self.molecule_mapping['molecules']:
            return self.molecule_mapping['molecules'][molecule]['aliases']
        return []