# KD-Text-Summarization


## 📁 Project Structure

```text
KD-Text-Summarization/
├── distillation/               # Code related to the distillation process
├── distilled-models-saved/     # Stores distilled (student) models after training
├── raw-datasets/               # Original input datasets used for summarization
├── student-models/             # Summarization evaluation results for student models
├── summarized-datasets/        # Summarized versions of the datasets
├── teacher-models/             # Summarization evaluation results for teacher models
├── venv/                       # Python virtual environment (excluded in .gitignore)
├── .gitignore
├── README.md
└── requirements.txt
```

## 🧠 Knowledge Distillation Workflow

The workflow includes:
- Running teacher models on raw datasets to generate summarized outputs.
- Training student models using the outputs of teacher models (distillation).
- Saving distilled student models for inference and evaluation.
- Comparing summarization quality between teacher and student models.

## 🏃 Execution Order

Please run the files in the following order:

### LLaMA 3.1 Models
- `KDTS_llama3.1_8b_260525.py`
- `KDTS_llama3.1_70b_260525.py`
- `KDTS_llama3.1_8b_distill_llama3.1_70b_260525.py`

### HuggingFace Export
- `Huggingface_exporter.py`  
  ⚠️ Make sure to configure paths, repo name, and HuggingFace credentials before running this script.

### Gemma 2 Models
- `KDTS_gemma2_2b_260525.py`
- `KDTS_gemma2_27b_260525.py`
- `KDTS_gemma2_2b_distill_gemma2_27b_260525.py`

### HuggingFace Export
- `Huggingface_exporter.py`  
  ⚠️ Make sure to configure paths, repo name, and HuggingFace credentials before running this script.

### Qwen 2.5 Models
- `KDTS_qwen2.5_7b_260525.py`
- `KDTS_qwen2.5_72b_260525.py`
- `KDTS_qwen2.5_7b_distill_qwen2.5_72b_260525.py`

### HuggingFace Export
- `Huggingface_exporter.py`  
  ⚠️ Make sure to configure paths, repo name, and HuggingFace credentials before running this script.

### Falcon Models
- `KDTS_falcon_7b_260525.py`
- `KDTS_falcon_40b_260525.py`
- `KDTS_falcon_7b_distill_falcon_40b_260525.py`


### HuggingFace Export
- `Huggingface_exporter.py`  
  ⚠️ Make sure to configure paths, repo name, and HuggingFace credentials before running this script.

## ✅ Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
