from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, login
import os


def upload_to_huggingface(model_dir, repo_id, readme_content, hf_token=None):
    """
    Upload a model and tokenizer saved in the same directory to Hugging Face Hub.
    """
    print(f"\n🚀 Preparing upload for: {repo_id}")

    # Optional: programmatic login (if not already logged in)
    if hf_token:
        login(token=hf_token)

    # Write model card
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Create Hugging Face repo
    create_repo(repo_id, exist_ok=True)

    # Load and push model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")


# ---------------------------------------
# ✅ CONFIGURE YOUR EXPORT HERE
# ---------------------------------------

model_dir = "../distilled-models-saved/gemma2-2b-distill-gemma2-27b"
repo_id = "your-username/gemma2-2b-distill-gemma2-27b"  # 🔁 CHANGE THIS TO YOUR HF USERNAME + REPO NAME
hf_token = None  # Optional: paste your Hugging Face token here if not using CLI login

# Description for README.md (model card)
readme = f"""
# 🧠 Distilled Gemma 2B from Gemma 2 27B

This model is distilled from `google/gemma-2-27b` to `google/gemma-2b` using KL-divergence on a custom summarization dataset.

## 🔧 Training Details
- Student Model: `google/gemma-2b`
- Teacher Model: `google/gemma-2-27b`
- Loss: KL Divergence
- Optimizer: AdamW
- Input Max Length: 1024 tokens
- Distillation Temperature: 2.0

## 📊 Evaluation Summary
Evaluation was done using:
- BLEU
- METEOR
- ROUGE-L
- BERTScore

## 🚀 Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "Summarize: The moon landing was a major event..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🏷️ Tags
distillation, summarization, gemma, causal-lm, huggingface-hub
"""

# Run the export
upload_to_huggingface(model_dir, repo_id, readme, hf_token)
