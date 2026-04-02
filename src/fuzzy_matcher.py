# src/fuzzy_matcher.py
from thefuzz import fuzz
from typing import List, Tuple, Optional, Dict

class FuzzyMatcher:
    """Fuzzy matching for molecule name inputs"""

    def __init__(self, molecule_mapping: Dict, threshold: int = 70):
        self.molecule_mapping = molecule_mapping
        self.threshold = threshold
        self.molecule_names = list(molecule_mapping.get("molecules", {}).keys())

    def _score_query(self, query: str, candidate: str) -> int:
        """Score a query against a single candidate string."""
        q, c = query.lower(), candidate.lower()
        if q == c:
            return 100
        score = max(fuzz.ratio(q, c), fuzz.partial_ratio(q, c))
        # Substring bonus
        if q in c or c in q:
            score = max(score, 60)
        return score

    def match_molecule_input(self, user_input: str) -> List[Tuple[str, int]]:
        """
        Find matching molecules based on user input.
        Returns a list of (molecule_name, confidence_score) tuples sorted by score.
        """
        if not user_input.strip():
            return []

        query = user_input.lower().strip()
        best_scores: Dict[str, int] = {}

        for mol_name, mol_data in self.molecule_mapping["molecules"].items():
            candidates = [mol_name] + mol_data.get("aliases", [])
            top_score = max(self._score_query(query, c) for c in candidates)
            if top_score >= self.threshold:
                best_scores[mol_name] = max(best_scores.get(mol_name, 0), top_score)

        return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

    def get_suggestions(self, user_input: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Return up to top_n molecule suggestions for the given input.
        Uses a lower threshold (40) and partial matching to be more forgiving.
        If no suggestions found, returns top_n molecules alphabetically (score=0).
        """
        if not user_input.strip():
            top_mols = sorted(self.molecule_mapping.get("molecules", {}).keys())[:top_n]
            return [(m, 0) for m in top_mols]

        query = user_input.lower().strip()
        suggestion_threshold = 40
        best_scores: Dict[str, int] = {}

        for mol_name, mol_data in self.molecule_mapping["molecules"].items():
            candidates = [mol_name] + mol_data.get("aliases", [])
            top_score = max(self._score_query(query, c) for c in candidates)
            if top_score >= suggestion_threshold:
                best_scores[mol_name] = max(best_scores.get(mol_name, 0), top_score)

        if not best_scores:
            # No suggestions found — return top_n alphabetically
            top_mols = sorted(self.molecule_mapping.get("molecules", {}).keys())[:top_n]
            return [(m, 0) for m in top_mols]

        sorted_suggestions = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:top_n]

    def get_top_match(self, user_input: str) -> Optional[str]:
        """Get the single best match above the configured threshold."""
        matches = self.match_molecule_input(user_input)
        return matches[0][0] if matches else None

    def get_aliases(self, molecule: str) -> List[str]:
        """Get all aliases for a molecule"""
        if molecule in self.molecule_mapping.get("molecules", {}):
            return self.molecule_mapping["molecules"][molecule].get("aliases", [])
        return []