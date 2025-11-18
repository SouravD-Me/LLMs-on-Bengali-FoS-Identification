# Bengali Figures of Speech — BengFoS 🌺

[![paper badge](https://img.shields.io/badge/Paper-PDF-blue)](paper/Bengali_FoS_Identification.pdf) [![license](https://img.shields.io/badge/License-CC--BY--NC--SA-lightgrey)](LICENSE)  
[![python](https://img.shields.io/badge/python-3.10%2B-green)]() [![status](https://img.shields.io/badge/status-experimental-orange)]()

---

## 🔎 Project snapshot

**Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification** — this repo contains the **BengFoS** dataset, reproducible code, visualization assets, and the paper PDF (`/paper`). The project evaluates LLMs (zero-shot & fine-tuned), probes hidden representations for figure-of-speech (FoS) signals, and explores deployment-friendly quantization strategies.

---

## Table of contents

- [Highlights](#-highlights)  
- [Dataset (BengFoS)](#-dataset-bengfos)  
- [Models & Key Results](#-models--key-results)  
- [Repository layout](#-repository-layout)  
- [Quickstart — run the experiments](#-quickstart---run-the-experiments)  
- [Results](#-results)  
- [Probing & interpretability](#-probing--interpretability)  
- [Figure of Speech — Label Map](#figure-of-speech--label-map)  
- [Citation](#-citation)  
- [Contact & Source Code Access](#-contact--source-code-access)  
- [License & Notes](#-license--notes)

---

# Highlights ✨

- **BengFoS dataset** — gold-standard, sentence-level annotations (≈3,148 annotated sentences) for 13 FoS categories.  
- **Experiments** — zero-shot baselines, full & parameter-efficient fine-tuning (Adapters / LoRA), quantized deployments (16-bit), 5-fold CV, and probing analyses.  
- **Interpretability** — layer-wise logistic probes and token-level attention visualizations that reveal where and how FoS is encoded.

---

# 📚 Dataset — BengFoS

- **Source:** Poems and lines from multiple Bengali poets (public-domain or cleared texts).  
- **Labels:** 13 classes (0..12) covering None, Simile, Metaphor, Personification, Onomatopoeia, Hyperbole, Alliteration, Oxymoron/Epigram, Irony, Euphemism/Pun, Apostrophe, Synecdoche/Metonymy, Assonance.  
- **Splits & preprocessing:** Stratified 5-fold CV; scripts perform Unicode normalization, tokenization, and optional augmentation/upsampling.

---

# 🧠 Models & Key Results (summary)

- Evaluated models include Llama-3 (8B), DeepSeek R1 Distill (7B), Mixtral (7B), and API baselines (GPT-3.5, Gemini-1.5).  
- Zero-shot results are weak, motivating fine-tuning; LoRA + 16-bit quantization provides strong trade-offs for deployment.

---

# 🗂 Repository layout (recommended)

├── README.md ├── LICENSE ├── paper/ │   └── Bengali_FoS_Identification.pdf ├── Visualizations/ │   ├── Probing Visualizations/ │   └── Deployment Visualizations/ ├── data/ ├── src/ ├── notebooks/ ├── experiments/ ├── results/ └── requirements.txt

---

# 🚀 Quickstart — run the experiments

Below is a focused Quickstart that includes data prep, optional upsampling/augmentation, training (LoRA & full), evaluation, quantization, and probing.

### 1) Create environment & install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2) Prepare data (tokenize, clean, create splits)

python src/utils.py \
  --prepare-data \
  --input data/raw \
  --output data/processed \
  --seq-length 128 \
  --seed 42

3) Optional — Upsampling / Augmentation (class balance)

python src/upsample.py \
  --input data/processed/train.jsonl \
  --output data/processed/train_upsampled.jsonl \
  --strategy class_balance \
  --target-min-count 500

Notes: --strategy may support class_balance, syn_aug, or backtranslate if implemented.

4) Train

LoRA (recommended for limited GPU memory)

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

Full fine-tuning

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

5-fold CV (example config)

python src/train.py --config experiments/cv_run.yaml

5) Evaluate (generate metrics & predictions)

python src/evaluate.py \
  --model_dir models/deepseek-lora \
  --test data/processed/test.jsonl \
  --out results/deepseek-lora-eval.json \
  --metrics micro_f1 macro_f1 per_label

6) Quantize & deploy (16-bit / 8-bit options)

python src/quantize.py --input models/deepseek-lora --bits 16 --output models/deepseek-lora-16bit
python src/evaluate.py --model_dir models/deepseek-lora-16bit --test data/processed/test.jsonl --out results/deepseek-lora-16bit-eval.json

Tip: LoRA + 16-bit quantization is a practical deployment choice.

7) Probing & interpretability (layer probes, attention maps)

python src/probe.py \
  --model_dir models/deepseek-lora \
  --data data/processed/test.jsonl \
  --layers all \
  --probe_out results/probes/ \
  --save_token_attentions Visualizations/Probing\ Visualizations/Attention_Examples/

Outputs: per-layer probe scores (CSV/JSON), and attention heatmaps in Visualizations/Probing Visualizations/.

8) Utilities

python src/plot_utils.py --probe_results results/probes/ --eval results/deepseek-lora-eval.json --out Visualizations/
python src/analysis_utils.py --pred results/deepseek-lora-eval.json --confusion results/confusion_deepseek.png


---

📊 Results

Our experiments reveal model limitations in zero-shot and how fine-tuning + quantization improves performance; probing further shows which layers encode FoS and which classes are easiest to separate.


---

Table 1 — Zero-shot comparison

Model	Accuracy	Avg. Confidence	F1 Score	Precision	Recall

Llama-3 8B	0.4211	0.4401	0.4178	0.4329	0.4016
DeepSeek-R1 Distill 7B	0.4179	0.4310	0.4023	0.4257	0.4175
Mixtral 7B	0.3536	0.3817	0.3410	0.3729	0.3386
GPT-3.5	0.3647	N/A	0.3538	0.3790	0.3472
Gemini-1.5	0.3818	N/A	0.3652	0.3812	0.3590


Table 1: Classification performance comparison of different LLMs on zero-shot setup. Confidence scores were not produced by API-based models (GPT/Gemini). Low zero-shot performance motivates adaptation via fine-tuning.


---

Table 2 — Fine-tuning performance (5-fold CV)

Model Variant	Accuracy	Macro-F1	Micro-F1

DeepSeek R1 (full)	0.55	0.53	0.56
DeepSeek R1 + Adapters	0.52	0.51	0.53
DeepSeek R1 + LoRA	0.51	0.50	0.52
DeepSeek R1 (16-bit quantized)	0.55	0.54	0.56
Llama-3 (full)	0.55	0.53	0.55
Llama-3 + Adapters	0.53	0.52	0.54
Llama-3 + LoRA	0.52	0.51	0.53
Llama-3 (16-bit quantized)	0.54	0.55	0.56


Table 2: Fine-tuning performance on BengFoS (5-fold CV) by both LLMs. The 16-bit quantized variants achieve marginally superior results.


---

Table 7 — Comparative deployment performance (16-bit quantized)

Metric	Llama-3 8B (Precision)	Llama-3 8B (Recall)	Llama-3 8B (F1)	Llama-3 8B (Support)	DeepSeek R1 7B (Precision)	DeepSeek R1 7B (Recall)	DeepSeek R1 7B (F1)	DeepSeek R1 7B (Support)

Micro Avg.	0.17	0.53	0.26	649	0.32	0.92	0.47	649
Macro Avg.	0.15	0.49	0.19	649	0.50	0.88	0.50	649
Weighted Avg.	0.35	0.53	0.40	649	0.58	0.92	0.64	649
Samples Avg.	0.16	0.52	0.24	649	0.58	0.90	0.64	649


Table 7: Comparative deployment performance of 16-bit quantized models on the BengFoS dataset.


---

🔬 Probing & interpretability

Layer signals: FoS cues peak in mid-layers for DeepSeek and later layers for Llama-3.

Class trends: Simile and Metaphor are easier to separate; subtler FoS types are harder.

Attention match: Tokens highlighted by attention align with probe results.

Deployment tip: Strong mid-layer signals suggest LoRA + 16-bit quantization is efficient.


Figure (placeholder):


Figure (placeholder):



---

Figure of Speech — Label Map

SL. No.	Figure of Speech	Class Label

1	None	0
2	Simile (উপমা)	1
3	Metaphor (রূপক)	2
4	Personification (মানবীকরণ)	3
5	Onomatopoeia (অনুকরণধ্বনি)	4
6	Hyperbole (অতিশয়োক্তি)	5
7	Alliteration (অনুপ্রাস)	6
8	Oxymoron and Antithesis, Epigram (বিরোধাভাস)	7
9	Irony (বিদ্রূপ)	8
10	Euphemism / Pun (শ্লেষ and যমক)	9
11	Apostrophe (উদ্দেশ্যোক্তি)	10
12	Synecdoche and Metonymy (প্রতিনিধিত্ব)	11
13	Assonance (স্বরবৈশিষ্ট্য)	12



---

📚 Citation

Please cite our paper if you use the dataset or code:

@inproceedings{das2025can,
  title={Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification},
  author={Das, Sourav and Ghosh, Kripabandhu},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={18645--18667},
  year={2025}
}


---

🤝 Contact & Source Code Access

Please contact us for the source codes. We will be happy to share.

Contact the authors (LinkedIn):

Sourav Das

Kripabandhu Ghosh


(Replace <sourav-linkedin-handle> and <kripabandhu-linkedin-handle> with your actual LinkedIn slugs.)


---

⚠️ License & Notes

Code: choose an appropriate open-source license (MIT / Apache 2.0 recommended).

Dataset: respect copyright for source poetry; follow the dataset license included in /data/README.md.

---
