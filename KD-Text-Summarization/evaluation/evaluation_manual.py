import os
import re
import pandas as pd
import nltk
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Download required NLTK resources (only once)
nltk.download("punkt")
nltk.download("wordnet")

# File paths
teacher_csv = "../raw-datasets/teacher_model_summaries_final.csv"
distilled_csv = "../raw-datasets/distilled_model_summaries_final.csv"
reference_txt = "../raw-datasets/text.txt"

# Read input files
teacher_df = pd.read_csv(teacher_csv)
distilled_df = pd.read_csv(distilled_csv)
with open(reference_txt, "r", encoding="utf-8") as f:
    ref_texts = f.read().strip().split("\n\n")

# Clean reference texts
ref_texts = [str(re.sub(r"^\d+\.\s*", "", text).strip()) for text in ref_texts]

# Extract summaries and ensure they are strings
teacher_summaries = teacher_df.iloc[:, 1].astype(str).tolist()
distilled_summaries = distilled_df.iloc[:, 1].astype(str).tolist()

# Trim all lists to the shortest length
min_len = min(len(ref_texts), len(teacher_summaries), len(distilled_summaries))
ref_texts = ref_texts[:min_len]
teacher_summaries = teacher_summaries[:min_len]
distilled_summaries = distilled_summaries[:min_len]

# Evaluation function (ROUGE, BLEU, METEOR only)
def compute_metrics(summaries, references):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_rouge, total_bleu, total_meteor = 0, 0, 0

    for ref, summ in zip(references, summaries):
        ref = str(ref).strip()
        summ = str(summ).strip()
        rouge = rouge_scorer_instance.score(ref, summ)['rougeL'].fmeasure
        bleu = sacrebleu.sentence_bleu(summ, [ref]).score
        meteor = meteor_score([ref.split()], summ.split())
        total_rouge += rouge
        total_bleu += bleu
        total_meteor += meteor

    return {
        "ROUGE-L": total_rouge / len(references),
        "BLEU": total_bleu / len(references),
        "METEOR": total_meteor / len(references)
    }

# Compute metrics
teacher_metrics = compute_metrics(teacher_summaries, ref_texts)
distilled_metrics = compute_metrics(distilled_summaries, ref_texts)

# Print results
print("\n[Teacher Model Metrics]")
for k, v in teacher_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n[Distilled Model Metrics]")
for k, v in distilled_metrics.items():
    print(f"{k}: {v:.4f}")
