import os
import csv
import re
import torch
import nltk
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from bert_score import score

# Ensure necessary NLTK data is downloaded
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

# Make directory if it doesn't exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Read lines from file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        phrases = file.readlines()
    return [phrase.strip() for phrase in phrases if phrase.strip()]

# Summarize text using HF model
def process_text_with_hf_model(pipeline_model, text):
    prompt = f"Summarize the following text:\n{text}"
    outputs = pipeline_model(prompt, max_new_tokens=200, do_sample=False)
    content = outputs[0]['generated_text'].replace(prompt, '')
    content = re.sub(r"^(here('|’)?s|this is|below is)?\s*(a\s*)?summary\s*(of\s*(the)?\s*text)?[:\-–]*\s*", "", content.strip(), flags=re.IGNORECASE)
    return content.strip()

# Evaluate text pairs
def evaluate_summary(original, summary):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = rouge.score(original, summary)['rougeL'].fmeasure
    bleu = sacrebleu.sentence_bleu(summary, [original]).score
    meteor = meteor_score([original.split()], summary.split())
    return rouge_score, bleu, meteor

# Save summary to CSV incrementally
def save_summary_to_csv(csv_path, original, summary):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Original Text", "Summary"])
        writer.writerow([original, summary])

# Save metrics to results.csv with dynamic column adjustment
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

# Main
def main():
    model_name = "google/gemma-2-2b"
    model_label = "gemma2_2b"
    input_file = "../raw-datasets/text.txt"
    summary_csv_path = f"../summarized-datasets/student-summarized/{model_label}_summarized.csv"
    metrics_csv_path = "../results/results.csv"

    print("[INFO] Downloading NLTK resources...")
    download_nltk_resources()

    print(f"[INFO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("[INFO] Ensuring output directories exist...")
    ensure_dir("../summarized-datasets/student-summarized/")
    ensure_dir("../results/")

    print(f"[INFO] Reading input texts from {input_file}")
    texts = read_text_file(input_file)

    summaries = []
    total_rouge, total_bleu, total_meteor = 0, 0, 0

    for idx, text in enumerate(texts, 1):
        print(f"[INFO] Summarizing text {idx}/{len(texts)}...")
        summary = process_text_with_hf_model(summarizer, text)
        summaries.append(summary)

        print("[INFO] Evaluating summary...")
        rouge_score, bleu_score, meteor_score_val = evaluate_summary(text, summary)

        total_rouge += rouge_score
        total_bleu += bleu_score
        total_meteor += meteor_score_val

        print("[INFO] Saving summary to CSV...")
        save_summary_to_csv(summary_csv_path, text, summary)

    avg_rouge = total_rouge / len(texts)
    avg_bleu = total_bleu / len(texts)
    avg_meteor = total_meteor / len(texts)

    print("[INFO] Computing BERTScore-F1...")
    P, R, F1 = score(summaries, texts, lang="en")

    print("\n[RESULTS] Evaluation Metrics for Student Model:gemma2_2b")
    print(f"Average ROUGE-L: {avg_rouge:.4f}")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"BERTScore-F1: {F1.mean():.4f}")

    print(f"[INFO] Saving evaluation results to {metrics_csv_path}")
    save_metrics_to_csv(metrics_csv_path, model_label, avg_rouge, avg_bleu, avg_meteor, F1.mean())

if __name__ == "__main__":
    main()
