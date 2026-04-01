# src/fuzzy_matcher.py
from fuzzywuzzy import fuzz
from typing import List, Tuple, Optional, Dict

class FuzzyMatcher:
    """Fuzzy matching for molecule name inputs"""
    
    def __init__(self, molecule_mapping: Dict, threshold: int = 70):
        self.molecule_mapping = molecule_mapping
        self.threshold = threshold

        # Cache molecule names for faster matching
        self.molecule_names = list(molecule_mapping.keys())

    
    
    def match_molecule_input(self, user_input: str) -> List[Tuple[str, int]]:
        """
        Find matching molecules based on user input.

        Returns a list of (molecule_name, confidence_score) tuples sorted by score.
        """
        if not user_input.strip():
            return []

        query = user_input.lower().strip()
        best_scores: Dict[str, int] = {}

        for mol_name, mol_data in self.molecule_mapping['molecules'].items():
            candidates = [mol_name] + mol_data.get('aliases', [])
            top_score = max(
                (100 if query == c.lower() else fuzz.ratio(query, c.lower()))
                for c in candidates
            )
            if top_score >= self.threshold:
                best_scores[mol_name] = max(best_scores.get(mol_name, 0), top_score)

        return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_top_match(self, user_input: str) -> Optional[str]:
        """Get the single best match"""
        matches = self.match_molecule_input(user_input)
        return matches[0][0] if matches else None
    
    def get_aliases(self, molecule: str) -> List[str]:
        """Get all aliases for a molecule"""
        if molecule in self.molecule_mapping['molecules']:
            return self.molecule_mapping['molecules'][molecule]['aliases']
        return []