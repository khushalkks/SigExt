## SigExt: Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization

This is the implementation of the EMNLP'24 paper with extended pipeline automation, hyperparameter tuning, and advanced deduplication features.

**Title:** [Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization](https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization)  
**Authors:** [Lei Xu](https://leixx.io/), [Asad Karim](https://www.amazon.science/author/asad-karim), [Saket Dingliwal](https://www.amazon.science/author/saket-dingliwal), [Aparna Elangovan](https://scholar.google.com/citations?user=eaow7uAAAAAJ&hl=en)  
**Venue:** EMNLP 2024 (Industry Track)

---

## 📌 Introduction

Large language models (LLMs) are highly effective at generating summaries across various domains through prompting techniques, reducing the need for dedicated training in summarization applications. However, designing prompts that guide LLMs to generate summaries with an appropriate level of detail and a coherent writing style can be challenging. 

**Keyphrase Signal Extractor (SigExt)** addresses this by leveraging salient information directly from the source document to improve summarization prompts. By integrating extracted keyphrases into LLM prompts, SigExt enhances ROUGE F1 and Recall, making generated summaries more aligned with reference texts.

---

## 🛠️ Key Features

- **Long-Context Keyphrase Extraction:** Fine-tunes and runs token classification on top of `allenai/longformer-large-4096` to process documents up to 4,096 tokens.
- **Steerable Prompt Engineering:** Injects keyphrase signals into custom prompting templates to direct LLMs to focus on core events.
- **Dynamic Parameter Tuning:** Automatically optimizes logits percentiles (e.g., [50, 60, 70, 75, 80, 85, 90]) and keyword limits (e.g., [5, 10, 15, 20, 25]) on the validation split using ROUGE overlap criteria before test execution.
- **Configurable Keyphrase Deduplication:** Suppresses redundant keyword candidates in prompts using `exact`, `substring`, or `fuzzy` Levenshtein-based filtering.
- **Local Model execution:** Fully interfaces with local LLMs (Mistral-7B via Ollama REST API) at zero API cost.

---

## 💡 Internship Contributions (Khushal Kumar Sahu)

During a 6-week summer research internship supervised by **Dr. Avinash Chandra Pandey** (Associate Professor, Department of Cyber Security and Digital Forensics, NFSU Dharwad Campus), Khushal Kumar Sahu designed, built, and evaluated several core components of this repository:

1. **Supervised Data Alignment Pipeline:** Designed the token mapping and scoring algorithm in `src/prepare_data.py` to automatically align source keyphrases with reference summary sentences using ROUGE overlap thresholds.
2. **Transformer Classifier Head:** Implemented and fine-tuned a PyTorch Lightning classifier head on top of the Longformer backbone to score entire candidate phrase spans.
3. **Dynamic Validation-Based Tuning:** Programmed the grid-search tuning loop in `src/zs_summarization.py` to automatically find optimal threshold percentiles and keyword limits without any GPU overhead.
4. **Automated Pipeline Orchestration:** Built the master execution script `run_pipeline.py` to handle Ollama health verification, automatic model pulling, inference caching, and evaluation scoring.

---

## ⚡ Quick Start: 1-Click Automated Pipeline

To run the end-to-end pipeline with model pings, Longformer inference, dynamic validation tuning, and local Mistral execution:

### Run with default settings:
```powershell
python run_pipeline.py
```

### Run with dynamic threshold tuning and substring deduplication:
```powershell
python run_pipeline.py --tune_threshold --dedup_strategy substring
```

---

## 🛠️ Step-by-Step Manual Execution

If you prefer to run individual pipeline stages manually:

### 1. Prepare Dataset
```powershell
python src/prepare_data.py --dataset cnn --output_dir experiments/cnn_dataset/
```

### 2. Train Longformer Extractor
```powershell
python src/train_longformer_extractor_context.py --dataset_dir experiments/cnn_dataset/ --checkpoint_dir experiments/cnn_extractor_model/
```

### 3. Keyphrase Extraction (Inference)
```powershell
python src/inference_longformer_extractor.py --dataset_dir experiments/cnn_dataset/ --checkpoint_dir experiments/cnn_extractor_model/ --output_dir experiments/cnn_dataset_with_keyphrase/
```

### 4. Zero-Shot Summarization & Evaluation
```powershell
python src/zs_summarization.py --model_name mistral --kw_strategy sigext_topk --kw_model_top_k 15 --dataset cnn --dataset_dir experiments/cnn_dataset_with_keyphrase/ --output_dir experiments/cnn_extsig_predictions/ --tune_threshold --dedup_strategy substring
```

---

## 📂 Codebase Structure

```text
SigExt/
├── run_pipeline.py              # Master execution pipeline & orchestrator
├── README.md                    # Project documentation
├── src/
│   ├── prepare_data.py          # Data downloading, RAKE extraction & tokenization
│   ├── train_longformer_extractor_context.py # Longformer model training
│   ├── inference_longformer_extractor.py     # Salient keyphrase extraction
│   ├── prompts.py               # Naive and Keyword-guided prompt templates
│   ├── zs_summarization.py      # Summarization generator & ROUGE evaluator
│   └── bedrock_utils.py         # Local Ollama REST API connector
└── experiments/
    ├── test_dataset/            # Sample pre-processed test dataset
    └── test_checkpoint/         # Pre-trained Longformer model weights
```

---

## 📊 Citation

```text
@inproceedings{xu2024salient,
  title={Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization},
  author={Xu, Lei and Karim, Mohammed Asad and Dingliwal, Saket and Elangovan, Aparna},
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
  year={2024}
}
```

## License

This project is licensed under the Apache-2.0 License.
