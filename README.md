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


## Execution Order


## ✅ Requirements
Install the necessary dependencies using:

pip install -r requirements.txt


Please run the files in the following order:


cd student-models

KDTS_llama3.1_8b_V2_160925.py



cd teacher-models

KDTS_llama3.1_70b_V2_160925.py



cd distillation

KDTS_llama3.1_8b_distill_llama3.1_70b_V2_160925.py







