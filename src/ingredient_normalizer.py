# src/ingredient_normalizer.py

LABEL_MAP = {
    "bell_pepper": "green pepper",
    "spring_onion": "green onions",
    "egg": "eggs",
    # extend as needed
}

def normalize(cv_labels: list[str]) -> list[str]:
    return [LABEL_MAP.get(label, label) for label in cv_labels]