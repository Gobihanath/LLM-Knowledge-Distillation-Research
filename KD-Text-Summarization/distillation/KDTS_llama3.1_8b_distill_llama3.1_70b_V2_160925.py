import os
import re
import csv
import torch
import nltk
import pandas as pd
import torch.nn.functional as F
from torch.optim import AdamW
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score
import evaluate

# ==============================
#  Setup and Utilities
# ==============================
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    nltk.download('punkt')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_input_file(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV file must contain 'id' and 'text' columns")
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.replace(r"\n?\d+\.\s*", "\n", regex=True)
    df["text"] = df["text"].str.replace(r"\n{2,}", "\n\n", regex=True)
    return df[["id", "text"]].values.tolist()

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def save_summary_to_csv(csv_path, id_val, original, summary):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "text", "summary"])
        writer.writerow([id_val, original, summary])

def save_metrics_to_csv(csv_path, model_name, rouge, bleu, meteor, bert_f1):
    today = datetime.now().strftime("%Y-%m-%d")
    new_row = {
        "Model": model_name,
        "Average ROUGE-L": f"{rouge:.4f}",
        "Average BLEU": f"{bleu:.4f}",
        "Average METEOR": f"{meteor:.4f}",
        "BERTScore-F1": f"{bert_f1:.4f}",
        "Date": today
    }

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        for col in new_row:
            if col not in existing_df.columns:
                existing_df[col] = None
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

# ==============================
#  Training & Evaluation Functions
# ==============================
def train_student_model(student, teacher, tokenizer, texts, lr=3e-5):
    optimizer = AdamW(student.parameters(), lr=lr)
    teacher.eval()
    student.train()
    losses = []

    for idx, (id_val, text) in enumerate(texts):
        prompt = f"Summarize:\n{text}\n\nSummary:"
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
            print(f"[TRAIN] Step {idx}/{len(texts)}, Loss: {loss.item():.4f}")

    return student

def evaluate_student_model(student, tokenizer, texts, summary_csv_path, metrics_csv_path, model_label):
    student.eval()
    generated_summaries = []
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    for idx, (id_val, text) in enumerate(texts):
        prompt = f"Summarize the following text:\n{text}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        output = student.generate(**inputs, max_new_tokens=256)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_summaries.append(summary)

        save_summary_to_csv(summary_csv_path, id_val, text, summary)
        print(f"[EVAL] Saved summary {idx + 1}/{len(texts)}")

    rouge_result = rouge.compute(predictions=generated_summaries, references=[t for _, t in texts])
    bleu_result = bleu.compute(predictions=generated_summaries, references=[t for _, t in texts])
    meteor_result = meteor.compute(predictions=generated_summaries, references=[t for _, t in texts])
    P, R, F1 = score(generated_summaries, [t for _, t in texts], lang="en")

    save_metrics_to_csv(metrics_csv_path, model_label,
                        rouge_result["rougeL"], bleu_result["bleu"],
                        meteor_result["meteor"], F1.mean())

    print("\n[RESULTS] Evaluation Metrics for Distilled Model")
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
    print(f"BLEU: {bleu_result['bleu']:.4f}")
    print(f"METEOR: {meteor_result['meteor']:.4f}")
    print(f"BERTScore-F1: {F1.mean():.4f}")

# ==============================
#  Main
# ==============================
def main():
    today = datetime.now().strftime("%Y-%m-%d")
    input_file = "../raw-datasets/text_30.csv"
    summary_csv_path = f"../summarized-datasets/distilled-model-summarized/llama3.1-8b-distill-llama3.1-70b_{today}.csv"
    metrics_csv_path = "../results/results.csv"
    model_save_path = f"../distilled-models-saved/llama3.1-8b-distill-llama3.1-70b_{today}"
    model_label = "llama3.1-8b-distill-llama3.1-70b"

    ensure_dir("../summarized-datasets/distilled-model-summarized/")
    ensure_dir("../results/")
    ensure_dir("../distilled-models-saved/")

    download_nltk_resources()

    print("[INFO] Loading and cleaning text file...")
    texts = clean_input_file(input_file)

    print("[INFO] Loading teacher and student models...")
    teacher_model_path = "meta-llama/Llama-3.1-70B"
    student_model_path = "meta-llama/Llama-3.1-8B"

    teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path, device_map="auto", torch_dtype=torch.float16)
    student = AutoModelForCausalLM.from_pretrained(student_model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Training student model via distillation...")
    student = train_student_model(student, teacher, tokenizer, texts)

    print("[INFO] Saving distilled model...")
    student.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print("[INFO] Evaluating distilled model...")
    evaluate_student_model(student, tokenizer, texts, summary_csv_path, metrics_csv_path, model_label)

if __name__ == "__main__":
    main()
