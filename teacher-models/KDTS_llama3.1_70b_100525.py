import subprocess
import shutil
import sys
import ollama
import nltk
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
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

# Check if Ollama CLI is installed
def is_ollama_installed():
    return shutil.which("ollama") is not None

# Warn user to install Ollama manually
def warn_ollama_manual_install():
    print("Ollama is not installed on this system.")
    print("Please install it manually from: https://ollama.com/download")
    sys.exit(1)

# Check if model is installed
def is_model_installed(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        print("Could not retrieve model list.")
        return False

# Pull model from Ollama
def pull_model(model_name):
    print(f"Pulling model: {model_name} ...")
    subprocess.run(["ollama", "pull", model_name], check=True)

# Setup Ollama and model
def setup_ollama_and_model(model_name):
    if not is_ollama_installed():
        warn_ollama_manual_install()
    else:
        print("Ollama is installed.")
    
    if not is_model_installed(model_name):
        print(f"Model '{model_name}' is not installed.")
        pull_model(model_name)
    else:
        print(f"Model '{model_name}' is already installed.")

# Read lines from file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        phrases = file.readlines()
    return [phrase.strip() for phrase in phrases if phrase.strip()]

# Generate summary with Ollama
def process_text_with_ollama(model_name, text):
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": f"Summarize the following text:\n{text}"}]
    )
    content = response["message"]["content"]

    # Remove common introductory summary phrases
    content = re.sub(r"^(here('|’)s|here is|this is|below is)?\s*(a\s*)?summary\s*(of\s*(the)?\s*text)?[:\-–]*\s*", "", content.strip(), flags=re.IGNORECASE)
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
    model_name = "llama3.1:70b"
    text_file_path = "../raw-datasets/text.txt"

    # Step 1: Download necessary NLTK data
    download_nltk_resources()

    # Step 2: Setup Ollama and Model
    setup_ollama_and_model(model_name)

    # Step 3: Read input texts
    texts = read_text_file(text_file_path)

    # Step 4: Summarize each text
    summaries = [process_text_with_ollama(model_name, text) for text in texts]

    # Step 5: Save generated summaries
    output_file = "../summarized-datasets/teacher-summarized/llama3.1_70b_summarized_texts.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        for summary in summaries:
            file.write(summary + "\n")

    # Step 6: Evaluate summaries
    avg_rougeL, avg_bleu, avg_meteor = evaluate_summaries(texts, summaries)

    # Step 7: Print Evaluation Results
    print("\nStudent Evaluation Results:")
    print(f"Average ROUGE-L Score: {avg_rougeL:.4f}")
    print(f"Average BLEU Score (SacreBLEU): {avg_bleu:.4f}")
    print(f"Average METEOR Score: {avg_meteor:.4f}")

# Run the script
if __name__ == "__main__":
    main()

