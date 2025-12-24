import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from rouge_score import rouge_scorer
from bert_score import score
import evaluate
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import re

# Download NLTK resources
nltk.download("wordnet")
nltk.download("punkt")

# Load teacher and student models
teacher_model_path = "google/gemma-2-27b"
student_model_path = "google/gemma-2-2b"

teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path, device_map="auto", torch_dtype=torch.float16)
student = AutoModelForCausalLM.from_pretrained(student_model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(student_model_path)

# ✅ Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Clean and process text
def clean_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = re.sub(r"\n?\d+\.\s*", "\n", raw_text.strip())
    cleaned_text = re.sub(r"\n{2,}", "\n\n", cleaned_text)
    texts = cleaned_text.strip().split("\n\n")
    return texts

texts = clean_text_file("../raw-datasets/text.txt")

# # Input length check (optional debug)
# lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
# print(f"Max tokens: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.2f}")

# Load reference summaries
with open("../summarized-datasets/teacher-summarized/gemma2_27b_summarized_texts.txt", "r", encoding="utf-8") as f:
    summaries = [s.strip() for s in f.readlines() if s.strip()]

dataset = Dataset.from_dict({"text": texts, "summary": summaries})

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

# Optimizer
optimizer = AdamW(student.parameters(), lr=3e-5)
teacher.eval()
student.train()

# Training loop
losses = []
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
        print(f"Step {idx}, Loss: {loss.item():.4f}")

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Distillation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Knowledge Distillation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save distilled student model
student.save_pretrained("../distilled-models-saved/gemma2-2b-distill-gemma2-27b")
tokenizer.save_pretrained("../distilled-models-saved/gemma2-2b-distill-gemma2-27b")

# Summarization function using student model
def summarize(text):
    prompt = f"Summarize:\n{text}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    output = student.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Evaluation
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

generated_summaries = []
for text in texts:
    gen_sum = summarize(text)
    generated_summaries.append(gen_sum)

bleu_result = bleu.compute(predictions=generated_summaries, references=summaries)
meteor_result = meteor.compute(predictions=generated_summaries, references=summaries)
rouge_result = rouge.compute(predictions=generated_summaries, references=summaries)
P, R, F1 = score(generated_summaries, summaries, lang="en")

# Print evaluation results
print("\n🔍 Evaluation Results-gemma2-2b-distill-gemma2-27b:")
print(f"BLEU: {bleu_result['bleu']:.4f}")
print(f"METEOR: {meteor_result['meteor']:.4f}")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
