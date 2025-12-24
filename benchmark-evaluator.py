# benchmark_evaluator.py

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def group_conversations(records: List[Dict]) -> List[Dict]:
    grouped = []
    temp = {}
    for item in records:
        if item['role'] == 'user':
            temp = {'question': item['content'], 'metadata': item['info']}
        elif item['role'] == 'assistant':
            temp['model_output'] = item['content']
            temp['answer_explained'] = item['info'].get('answer_content', '')
            grouped.append(temp)
            temp = {}
    return grouped


def evaluate_accuracy(conversations: List[Dict]) -> float:
    correct = 0
    total = 0
    for entry in conversations:
        ref_answer = entry['metadata'].get('reference_answer')
        model_output = entry['model_output']

        if not ref_answer:
            continue

        if f"\boxed{{{ref_answer}}}" in model_output or ref_answer in model_output:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_length_stats(conversations: List[Dict]) -> Dict:
    lengths = [len(entry['model_output'].split()) for entry in conversations]
    return {
        'min': min(lengths),
        'max': max(lengths),
        'average': sum(lengths) / len(lengths) if lengths else 0
    }


def main():
    input_file = 'am_0.9M_sample_1k.jsonl' 

    print(f"Loading dataset from {input_file}...")
    records = load_jsonl(input_file)

    print("Grouping question-answer pairs...")
    conversations = group_conversations(records)

    print("Evaluating accuracy...")
    acc = evaluate_accuracy(conversations)
    print(f"\nAccuracy: {acc * 100:.2f}%")

    print("\nCalculating output length statistics...")
    length_stats = evaluate_length_stats(conversations)
    print("Length Stats:", length_stats)


if __name__ == '__main__':
    main()
