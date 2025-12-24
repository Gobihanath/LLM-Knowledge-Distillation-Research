import json
import os
from evaluation import evaluate_example
import csv

if __name__ == "__main__":
    data_dir = "../datasets/processed"
    result_path = "../results/benchmark_results.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Question", "Reference Answer", "Predicted Answer", "Accuracy"])

        for file in os.listdir(data_dir):
            if not file.endswith(".jsonl"): continue
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    question = example['question']
                    gold = example['answer']
                    prediction = "<LLM_CALL>"  # Replace this with actual model call
                    acc = evaluate_example(gold, prediction)
                    writer.writerow([example['dataset'], question, gold, prediction, acc])

