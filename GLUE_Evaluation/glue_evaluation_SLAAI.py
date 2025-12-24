import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# -------------------------
# 1. Load CSV files
# -------------------------
text_df = pd.read_csv("../raw-datasets/text_30.csv")  # columns: id,text
distilled_df = pd.read_csv("../raw-datasets/distilled_model_summaries_final.csv")  # columns: ID,Distilled Model Summary
teacher_df = pd.read_csv("../raw-datasets/teacher_model_summaries_final.csv")      # columns: ID,Teacher Model Summary

# Ensure consistent column names
text_df.rename(columns={"id": "ID", "text": "Text"}, inplace=True)
distilled_df.rename(columns=lambda x: x.strip(), inplace=True)
teacher_df.rename(columns=lambda x: x.strip(), inplace=True)

# Merge into one DataFrame
df = text_df.merge(teacher_df, on="ID").merge(distilled_df, on="ID")

# -------------------------
# 2. Load MNLI Model
# -------------------------
mnli_model_name = "roberta-large-mnli"
mnli_tokenizer = AutoTokenizer.from_pretrained(mnli_model_name)
mnli_model = AutoModelForSequenceClassification.from_pretrained(mnli_model_name)
mnli_model.eval()

mnli_labels = {0: "contradiction", 1: "neutral", 2: "entailment"}

def mnli_entailment(premise, hypothesis):
    inputs = mnli_tokenizer(hypothesis, premise, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = mnli_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return mnli_labels[pred]

# -------------------------
# 3. Load STS-B Model (for embeddings)
# -------------------------
sts_model_name = "sentence-transformers/stsb-roberta-large"
sts_tokenizer = AutoTokenizer.from_pretrained(sts_model_name)
sts_model = AutoModel.from_pretrained(sts_model_name)  # AutoModel to get embeddings
sts_model.eval()

def sts_similarity(text1, text2):
    inputs1 = sts_tokenizer(text1, return_tensors="pt", truncation=True)
    inputs2 = sts_tokenizer(text2, return_tensors="pt", truncation=True)
    with torch.no_grad():
        emb1 = sts_model(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = sts_model(**inputs2).last_hidden_state.mean(dim=1)
    return F.cosine_similarity(emb1, emb2).item()

# -------------------------
# 4. Evaluate Teacher & Distilled Summaries
# -------------------------
teacher_entailment_results = []
distilled_entailment_results = []
teacher_sts_scores = []
distilled_sts_scores = []

for _, row in df.iterrows():
    original_text = row["Text"]
    teacher_sum = row["Teacher Model Summary"]
    distilled_sum = row["Distilled Model Summary"]

    # MNLI entailment check
    teacher_entailment_results.append(mnli_entailment(original_text, teacher_sum))
    distilled_entailment_results.append(mnli_entailment(original_text, distilled_sum))

    # STS-B cosine similarity
    teacher_sts_scores.append(sts_similarity(original_text, teacher_sum))
    distilled_sts_scores.append(sts_similarity(original_text, distilled_sum))

# Add results to DataFrame
df["Teacher_MNLI"] = teacher_entailment_results
df["Distilled_MNLI"] = distilled_entailment_results
df["Teacher_STS"] = teacher_sts_scores
df["Distilled_STS"] = distilled_sts_scores

# -------------------------
# 5. Print averages
# -------------------------
teacher_entail_acc = teacher_entailment_results.count("entailment") / len(teacher_entailment_results)
distilled_entail_acc = distilled_entailment_results.count("entailment") / len(distilled_entailment_results)

print(f"Teacher MNLI Entailment Accuracy: {teacher_entail_acc:.2f}")
print(f"Distilled MNLI Entailment Accuracy: {distilled_entail_acc:.2f}")
print(f"Teacher STS-B Avg Similarity: {np.mean(teacher_sts_scores):.4f}")
print(f"Distilled STS-B Avg Similarity: {np.mean(distilled_sts_scores):.4f}")

# -------------------------
# 6. Save results
# -------------------------
df.to_csv("../raw-datasets/evaluation_results_with_original.csv", index=False)
print("Evaluation results saved to ../raw-datasets/evaluation_results_with_original.csv")
