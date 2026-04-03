import json
from pathlib import Path
from typing import List, Dict

DATASET_PATH = Path(__file__).parent / "ragas_evaluation_dataset.json"

def load_golden_dataset() -> List[Dict]:
    """Load the golden evaluation dataset from JSON."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} golden evaluation examples")
    return data


def get_evaluation_questions() -> List[str]:
    """Return list of questions only (for quick testing)."""
    dataset = load_golden_dataset()
    return [item["question"] for item in dataset]