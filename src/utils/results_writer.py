"""
Results Serialization Utility
"""

import json
import pickle
import os
from datetime import datetime


class ResultsWriter:
    """A simple class to write results to a file."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []

    def add(self, result: dict):
        """Add a result to the collection."""
        self.results.append(result)

    def add_result(self, result: dict):
        """Alias for add() for compatibility."""
        self.add(result)

    def write(self):
        """Write results to JSON and pickle files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Write to JSON
        json_path = os.path.join(self.output_dir, f"results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=4)

        # Write to pickle
        pkl_path = os.path.join(self.output_dir, f"results_{timestamp}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.results, f)
        
        return json_path

    def save(self):
        """Alias for write() for compatibility."""
        return self.write()
