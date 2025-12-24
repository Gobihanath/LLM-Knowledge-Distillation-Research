import json
import os

def normalize_dataset(dataset_path, output_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    normalized = []
    for line in lines:
        example = json.loads(line)
        normalized.append({
            'question': example.get('content', ''),
            'answer': example.get('info', {}).get('reference_answer', ''),
            'dataset': os.path.basename(dataset_path).split('.')[0]
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in normalized:
            f.write(json.dumps(ex) + '\n')

if __name__ == "__main__":
    input_dir = "../datasets"
    output_dir = "../datasets/processed"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".jsonl"):
            normalize_dataset(os.path.join(input_dir, file), os.path.join(output_dir, file))
