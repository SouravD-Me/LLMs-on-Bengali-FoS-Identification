# 🌺 [Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification](https://aclanthology.org/2025.emnlp-main.941/)

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue?style=flat-square)](paper/Bengali_FoS_Identification.pdf)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA-lightgrey?style=flat-square)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-green?style=flat-square)]()
[![Status](https://img.shields.io/badge/status-experimental-orange?style=flat-square)]()

---

## 🔍 Project Snapshot

This repository contains the **BengFoS dataset** from our EMNLP 2025 (Main Conference) paper. Our project evaluates LLMs (zero-shot & fine-tuned), probes hidden representations for figure-of-speech (FoS) signals, and explores deployment-friendly quantization strategies for Bengali literary analysis.

---

## 📋 Table of Contents

- [✨ Highlights](#-highlights)
- [📊 Dataset](#-dataset)
- [🤖 Models & Key Results](#-models--key-results)
- [📁 Repository Layout](#-repository-layout)
- [🚀 Quickstart](#-quickstart)
- [📈 Results](#-results)
- [🔍 Probing & Interpretability](#-probing--interpretability)
- [🏷️ Figure of Speech Label Map](#️-figure-of-speech-label-map)
- [📚 Citation](#-citation)
- [📞 Contact](#-contact)
- [📄 License](#-license)

---

## ✨ Highlights

- **BengFoS Dataset**: Gold-standard, sentence-level annotations (≈3,148 annotated sentences) for 13 FoS categories
- **Comprehensive Experiments**: Zero-shot baselines, full & parameter-efficient fine-tuning (Adapters / LoRA), quantized deployments (16-bit), 5-fold CV, and probing analyses
- **Interpretability Insights**: Layer-wise logistic probes and token-level attention visualizations that reveal where and how FoS is encoded in LLM representations

---

## 📊 Dataset — BengFoS

| **Attribute** | **Details** |
|---------------|-------------|
| **Source** | Poems and lines from multiple Bengali poets (public-domain or cleared texts) |
| **Labels** | 13 classes (0-12): None, Simile, Metaphor, Personification, Onomatopoeia, Hyperbole, Alliteration, Oxymoron/Epigram, Irony, Euphemism/Pun, Apostrophe, Synecdoche/Metonymy, Assonance |
| **Splits & Preprocessing** | Stratified 5-fold CV; Unicode normalization, tokenization, and optional augmentation/upsampling |

---

## 🤖 Models & Key Results

### Evaluated Models
- **Open-source LLMs**: Llama-3 (8B), DeepSeek R1 Distill (7B), Mixtral (7B)
- **API Baselines**: GPT-3.5, Gemini-1.5

### Key Findings
- **Zero-shot performance** is weak across all models, motivating fine-tuning approaches
- **LoRA + 16-bit quantization** provides the best trade-offs for practical deployment
- **Parameter-efficient methods** (LoRA/Adapters) achieve competitive results with significantly reduced computational overhead

---

## 📁 Repository Layout

```bash
├── README.md                  # Documentation (this file)
├── LICENSE                    # Project license
├── paper/
│   └── Bengali_FoS_Identification.pdf  # Research paper
├── Visualizations/            # Generated visualizations
│   ├── Probing_Visualizations/  # Layer-wise probing results
│   └── Deployment_Visualizations/  # Model comparison charts
├── data/                      # Dataset files
│   └── README.md              # Data license and usage guidelines
├── src/                       # Source code
├── notebooks/                 # Jupyter notebooks for analysis
├── experiments/               # Experiment configurations
├── results/                   # Evaluation results and metrics
└── requirements.txt           # Python dependencies
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python src/utils.py \
  --prepare-data \
  --input data/raw \
  --output data/processed \
  --seq-length 128 \
  --seed 42
```

### 3. Optional: Class Balancing
```bash
python src/upsample.py \
  --input data/processed/train.jsonl \
  --output data/processed/train_upsampled.jsonl \
  --strategy class_balance \
  --target-min-count 500
```

### 4. Model Training

#### LoRA Fine-tuning (Recommended)
```bash
python src/train.py \
  --model deepseek-r1-7b \
  --train data/processed/train_upsampled.jsonl \
  --dev data/processed/dev.jsonl \
  --output_dir models/deepseek-lora \
  --method lora \
  --lora_rank 8 \
  --batch_size 16 \
  --lr 3e-5 \
  --epochs 6 \
  --max_seq_length 128 \
  --fp16
```

#### Full Fine-tuning
```bash
python src/train.py \
  --model deepseek-r1-7b \
  --train data/processed/train_upsampled.jsonl \
  --dev data/processed/dev.jsonl \
  --output_dir models/deepseek-ft \
  --method full \
  --batch_size 8 \
  --lr 2e-5 \
  --epochs 6 \
  --max_seq_length 128 \
  --fp16
```

### 5. Evaluation
```bash
python src/evaluate.py \
  --model_dir models/deepseek-lora \
  --test data/processed/test.jsonl \
  --out results/deepseek-lora-eval.json \
  --metrics micro_f1 macro_f1 per_label
```

### 6. Quantization & Deployment
```bash
python src/quantize.py --input models/deepseek-lora --bits 16 --output models/deepseek-lora-16bit
python src/evaluate.py --model_dir models/deepseek-lora-16bit --test data/processed/test.jsonl --out results/deepseek-lora-16bit-eval.json
```

### 7. Probing Analysis
```bash
python src/probe.py \
  --model_dir models/deepseek-lora \
  --data data/processed/test.jsonl \
  --layers all \
  --probe_out results/probes/ \
  --save_token_attentions "Visualizations/Probing_Visualizations/Attention_Examples/"
```

---

## 📈 Results

### Table 1: Zero-Shot Comparison
| Model | Accuracy | Avg. Confidence | F1 Score | Precision | Recall |
|-------|----------|-----------------|----------|-----------|--------|
| Llama-3 8B | 0.4211 | 0.4401 | 0.4178 | 0.4329 | 0.4016 |
| DeepSeek-R1 Distill 7B | 0.4179 | 0.4310 | 0.4023 | 0.4257 | 0.4175 |
| Mixtral 7B | 0.3536 | 0.3817 | 0.3410 | 0.3729 | 0.3386 |
| GPT-3.5 | 0.3647 | N/A | 0.3538 | 0.3790 | 0.3472 |
| Gemini-1.5 | 0.3818 | N/A | 0.3652 | 0.3812 | 0.3590 |

> **Note**: Low zero-shot performance motivates adaptation via fine-tuning. Confidence scores not available for API-based models.

### Table 2: Fine-Tuning Performance (5-Fold CV)
| Model Variant | Accuracy | Macro-F1 | Micro-F1 |
|---------------|----------|----------|----------|
| DeepSeek R1 (full) | 0.55 | 0.53 | 0.56 |
| DeepSeek R1 + Adapters | 0.52 | 0.51 | 0.53 |
| DeepSeek R1 + LoRA | 0.51 | 0.50 | 0.52 |
| DeepSeek R1 (16-bit quantized) | 0.55 | 0.54 | 0.56 |
| Llama-3 (full) | 0.55 | 0.53 | 0.55 |
| Llama-3 + Adapters | 0.53 | 0.52 | 0.54 |
| Llama-3 + LoRA | 0.52 | 0.51 | 0.53 |
| Llama-3 (16-bit quantized) | 0.54 | 0.55 | 0.56 |

### Table 3: Comparative Deployment Performance (16-bit Quantized)
| Metric | Llama-3 8B (Precision) | Llama-3 8B (Recall) | Llama-3 8B (F1) | DeepSeek R1 7B (Precision) | DeepSeek R1 7B (Recall) | DeepSeek R1 7B (F1) |
|--------|------------------------|---------------------|-----------------|----------------------------|--------------------------|---------------------|
| Micro Avg. | 0.17 | 0.53 | 0.26 | 0.32 | 0.92 | 0.47 |
| Macro Avg. | 0.15 | 0.49 | 0.19 | 0.50 | 0.88 | 0.50 |
| Weighted Avg. | 0.35 | 0.53 | 0.40 | 0.58 | 0.92 | 0.64 |
| Samples Avg. | 0.16 | 0.52 | 0.24 | 0.58 | 0.90 | 0.64 |

---

## 🔍 Probing & Interpretability

### Key Insights
- **Layer Signals**: FoS cues peak in mid-layers for DeepSeek and later layers for Llama-3
- **Class Trends**: Simile and Metaphor are easier to separate; subtler FoS types (Irony, Euphemism) are harder
- **Attention Alignment**: Tokens highlighted by attention mechanisms align with probe results
- **Deployment Strategy**: Strong mid-layer signals suggest LoRA + 16-bit quantization is highly efficient

### Visualization Examples
![Layer-wise FoS Signal Probing](https://via.placeholder.com/600x400?text=Probing+Visualization+-+Layer+Signals)
*Layer-wise FoS signal probing results across transformer layers*

![Token-level Attention Heatmaps](https://via.placeholder.com/600x400?text=Attention+Visualization+-+Token+Level)
*Token-level attention heatmaps for FoS identification*

---

## 🏷️ Figure of Speech — Label Map

| SL. No. | Figure of Speech (English) | Figure of Speech (Bengali) | Class Label |
|---------|----------------------------|----------------------------|-------------|
| 1 | None | - | 0 |
| 2 | Simile | উপমা | 1 |
| 3 | Metaphor | রূপক | 2 |
| 4 | Personification | মানবীকরণ | 3 |
| 5 | Onomatopoeia | অনুকরণধ্বনি | 4 |
| 6 | Hyperbole | অতিশয়োক্তি | 5 |
| 7 | Alliteration | অনুপ্রাস | 6 |
| 8 | Oxymoron/Antithesis/Epigram | বিরোধাভাস | 7 |
| 9 | Irony | বিদ্রূপ | 8 |
| 10 | Euphemism/Pun | শ্লেষ/যমক | 9 |
| 11 | Apostrophe | উদ্দেশ্যোক্তি | 10 |
| 12 | Synecdoche/Metonymy | প্রতিনিধিত্ব | 11 |
| 13 | Assonance | স্বরবৈশিষ্ট্য | 12 |

---

## 📚 Citation

If you use the BengFoS dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{das2025can,
  title={Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification},
  author={Das, Sourav and Ghosh, Kripabandhu},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={18645--18667},
  year={2025}
}
```

---

## 📞 Contact

### **Note**: Please contact us for the complete codebase of the project. We will be happy to share the same.

- **[Sourav Das](https://www.linkedin.com/in/sourav-das/)** 
- **[Kripabandhu Ghosh](https://www.linkedin.com/in/kripabandhu-ghosh/)**

---

## 📄 License

| Component | License | Details |
|-----------|---------|---------|
| **Dataset** | Custom License | Respect copyright for source poetry; follow dataset license in [data/README.md](data/README.md) |
| **Paper** | CC-BY-NC-SA | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International |

---

**Last Updated**: November 18, 2025  

```
