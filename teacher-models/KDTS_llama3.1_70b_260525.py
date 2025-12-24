import subprocess
import shutil
import sys
import nltk
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# Download NLTK resources only if not already available
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

# Read lines from file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        phrases = file.readlines()
    return [phrase.strip() for phrase in phrases if phrase.strip()]

# Generate summary using HuggingFace model
def process_text_with_hf_model(pipeline_model, text):
    prompt = f"Summarize the following text:\n{text}"
    outputs = pipeline_model(prompt, max_new_tokens=200, do_sample=False)
    content = outputs[0]['generated_text'].replace(prompt, '')

    # Clean up common summary prefixes
    content = re.sub(r"^(here('|’)?s|this is|below is)?\s*(a\s*)?summary\s*(of\s*(the)?\s*text)?[:\-–]*\s*", "", content.strip(), flags=re.IGNORECASE)
    return content.strip()

# Evaluate summaries
def evaluate_summaries(original_texts, summaries):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_rougeL, total_bleu, total_meteor = 0, 0, 0
    num_samples = len(original_texts)

    for original, summary in zip(original_texts, summaries):
        rouge_scores = rouge.score(original, summary)
        bleu = sacrebleu.sentence_bleu(summary, [original])
        meteor = meteor_score([original.split()], summary.split())

        total_rougeL += rouge_scores['rougeL'].fmeasure
        total_bleu += bleu.score
        total_meteor += meteor

    avg_rougeL = total_rougeL / num_samples
    avg_bleu = total_bleu / num_samples
    avg_meteor = total_meteor / num_samples

    return avg_rougeL, avg_bleu, avg_meteor

# Main function
def main():
    model_name = "meta-llama/Llama-3.1-70B"
    text_file_path = "../raw-datasets/text.txt"
    output_file = "../summarized-datasets/teacher-summarized/llama3.1_70b_summarized_texts.txt"

    # Step 1: Download necessary NLTK data
    download_nltk_resources()

    # Step 2: Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Step 3: Create summarization pipeline
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Step 4: Read input texts
    texts = read_text_file(text_file_path)

    # Step 5: Summarize each text
    summaries = [process_text_with_hf_model(summarizer, text) for text in texts]

    # Step 6: Save generated summaries
    with open(output_file, "w", encoding="utf-8") as file:
        for summary in summaries:
            file.write(summary + "\n")

    # Step 7: Evaluate summaries
    avg_rougeL, avg_bleu, avg_meteor = evaluate_summaries(texts, summaries)

    # Step 8: Print Evaluation Results
    print("\nTeacher Evaluation Results_Llama3.1_70b:")
    print(f"Average ROUGE-L Score: {avg_rougeL:.4f}")
    print(f"Average BLEU Score (SacreBLEU): {avg_bleu:.4f}")
    print(f"Average METEOR Score: {avg_meteor:.4f}")

# Run the script
if __name__ == "__main__":
    main()
