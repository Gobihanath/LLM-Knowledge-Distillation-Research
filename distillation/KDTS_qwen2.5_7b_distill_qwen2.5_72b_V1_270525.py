import os
import re
import torch
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score
import evaluate

# Download NLTK resources
nltk.download("wordnet")
nltk.download("punkt")

# Helper Functions
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = re.sub(r"\n?\d+\.\s*", "\n", raw_text.strip())
    cleaned_text = re.sub(r"\n{2,}", "\n\n", cleaned_text)
    texts = cleaned_text.strip().split("\n\n")
    return texts

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def save_summary_to_csv(csv_path, original, summary):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        df = pd.DataFrame([[original, summary]], columns=["Original Text", "Summary"])
        df.to_csv(f, header=write_header, index=False)

def save_metrics_to_csv(csv_path, model_name, rouge, bleu, meteor, bert_f1):
    new_row = {
        "Model": model_name,
        "Average ROUGE-L": f"{rouge:.4f}",
        "Average BLEU": f"{bleu:.4f}",
        "Average METEOR": f"{meteor:.4f}",
        "BERTScore-F1": f"{bert_f1:.4f}"
    }

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)

        # Add missing columns if needed
        for col in new_row:
            if col not in existing_df.columns:
                existing_df[col] = None

        # Ensure all required columns are in both dataframes
        for col in existing_df.columns:
            if col not in new_row:
                new_row[col] = None

        new_df = pd.DataFrame([new_row])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        print(f"[INFO] Appended metrics for {model_name} to existing file.")
    else:
        pd.DataFrame([new_row]).to_csv(csv_path, index=False)
        print(f"[INFO] Created new metrics file with {model_name}.")

# File Paths
text_path = "../raw-datasets/text.txt"
summary_path = "../summarized-datasets/distilled-model-summarized/qwen2.5-7b-distill-qwen2.5-72b.csv"
metrics_path = "../results/results.csv"
model_save_path = "../distilled-models-saved/qwen2.5-7b-distill-qwen2.5-72b"
model_label = "qwen2.5-7b-distill-qwen2.5-72b"

# Ensure directories
ensure_dir("../summarized-datasets/distilled-model-summarized/")
ensure_dir("../results/")
ensure_dir("../distilled-models-saved/")


teacher_model_path = "Qwen/Qwen2.5-72B"
student_model_path = "Qwen/Qwen2.5-7B"

# Load texts
print("[INFO] Loading and cleaning text file...")
texts = clean_text_file(text_path)

# Load models
print("[INFO] Loading teacher and student models...")
teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path, device_map="auto", torch_dtype=torch.float16)
student = AutoModelForCausalLM.from_pretrained(student_model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(student_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Optimizer
optimizer = AdamW(student.parameters(), lr=3e-5)
teacher.eval()
student.train()

# Prepare dataset
dataset = Dataset.from_dict({"text": texts})
losses = []

# 🔁 Distillation loop
for idx, sample in enumerate(dataset):
    prompt = f"Summarize:\n{sample['text']}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits

    student_logits = student(**inputs).logits
    loss = distillation_loss(student_logits, teacher_logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if idx % 5 == 0:
        print(f"[TRAIN] Step {idx}/{len(dataset)}, Loss: {loss.item():.4f}")

# #  Plot loss
# plt.figure(figsize=(10, 5))
# plt.plot(losses, label="Distillation Loss")
# plt.xlabel("Training Step")
# plt.ylabel("Loss")
# plt.title("Distillation Loss Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 💾 Save student model
print("[INFO] Saving distilled model...")
student.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# 🔍 Evaluation phase
print("[INFO] Running summarization with distilled model...")
student.eval()
generated_summaries = []
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

for idx, text in enumerate(texts):
    prompt = f"Summarize:\n{text}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    output = student.generate(**inputs, max_new_tokens=256)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_summaries.append(summary)
    save_summary_to_csv(summary_path, text, summary)
    print(f"[EVAL] Saved summary {idx + 1}/{len(texts)}")

# 📊 Compute metrics
print("[INFO] Computing evaluation metrics...")
rouge_result = rouge.compute(predictions=generated_summaries, references=texts)
bleu_result = bleu.compute(predictions=generated_summaries, references=texts)
meteor_result = meteor.compute(predictions=generated_summaries, references=texts)
P, R, F1 = score(generated_summaries, texts, lang="en")

# ✅ Save metrics
print("[INFO] Saving evaluation results...")
save_metrics_to_csv(metrics_path, model_label, rouge_result["rougeL"], bleu_result["bleu"], meteor_result["meteor"], F1.mean())

# 📢 Print results
print("\n📌 Evaluation Results:qwen2.5-7b-distill-qwen2.5-72b")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"BLEU: {bleu_result['bleu']:.4f}")
print(f"METEOR: {meteor_result['meteor']:.4f}")
print(f"BERTScore-F1: {F1.mean():.4f}")
