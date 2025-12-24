def evaluate_example(reference: str, prediction: str) -> float:
    reference = reference.strip().lower()
    prediction = prediction.strip().lower()
    return float(reference == prediction)


