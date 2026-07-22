## SigExt: Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization

This is the implementation of the EMNLP'24 paper.

**Title:** [Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization](https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization)  
**Authors:** [Lei Xu](https://leixx.io/), [Asad Karim](https://www.amazon.science/author/asad-karim), [Saket Dingliwal](https://www.amazon.science/author/saket-dingliwal), [Aparna Elangovan](https://scholar.google.com/citations?user=eaow7uAAAAAJ&hl=en)  
**Venue:** EMNLP 2024 (Industry Track)

---

## 📌 Introduction

Large language models (LLMs) are highly effective at generating summaries across various domains through prompting techniques, reducing the need for dedicated training in summarization applications. However, designing prompts that guide LLMs to generate summaries with an appropriate level of detail and a coherent writing style can be challenging. 

**Keyphrase Signal Extractor (SigExt)** addresses this by leveraging salient information directly from the source document to improve summarization prompts. By integrating extracted keyphrases into LLM prompts, SigExt enhances ROUGE F1 and Recall, making generated summaries more aligned with reference texts.

---

## ⚡ Quick Start: 1-Click Automated Pipeline

This updated repository includes an automated master orchestrator (`run_pipeline.py`) that handles Ollama health checks, model inference, zero-shot summarization, and ROUGE metric calculations in a single command.

### Prerequisites:
1. Ensure Python 3.8+ and required libraries (`transformers`, `torch`, `rouge-score`, `rapidfuzz`, `requests`) are installed.
2. Ensure **Ollama** is running locally on your system (`http://localhost:11434`).

### Run the Pipeline:
```powershell
python run_pipeline.py
```

### What `run_pipeline.py` executes automatically:
1. **Ollama Readiness Check:** Pings Ollama server and pulls the `mistral` model if missing.
2. **Stage 1 (Keyphrase Extraction):** Uses Longformer (`allenai/longformer-large-4096`) to extract salient keyphrases.
3. **Stage 2 (Summarization & Evaluation):** Injects keyphrases into zero-shot prompts, queries local Mistral LLM, and calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics.

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
python src/zs_summarization.py --model_name mistral --kw_strategy sigext_topk --kw_model_top_k 15 --dataset cnn --dataset_dir experiments/cnn_dataset_with_keyphrase/ --output_dir experiments/cnn_extsig_predictions/
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


