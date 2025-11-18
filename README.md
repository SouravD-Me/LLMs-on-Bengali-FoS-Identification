# Bengali Figures of Speech — BengFoS 🌺

[![paper badge](https://img.shields.io/badge/Paper-PDF-blue)](paper/Bengali_FoS_Identification.pdf)
[![license](https://img.shields.io/badge/License-CC--BY--NC--SA-lightgrey)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![status](https://img.shields.io/badge/status-experimental-orange)]()

---

## 🔎 Project Snapshot

**Can LLMs be Literary Companions?: Analysing LLMs on Bengali Figures of Speech Identification** — this repo contains the **BengFoS** dataset, reproducible code, visualization assets, and the paper PDF (`/paper`). The project evaluates LLMs (zero-shot & fine-tuned), probes hidden representations for figure-of-speech (FoS) signals, and explores deployment-friendly quantization strategies.

---

## Table of Contents

- [Highlights](#highlights-)
- [Dataset](#dataset--bengfos)
- [Models & Key Results](#models--key-results)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart--run-the-experiments)
- [Results](#results)
- [Probing & Interpretability](#probing--interpretability)
- [Figure of Speech Label Map](#figure-of-speech--label-map)
- [Citation](#citation)
- [Contact & Source Code Access](#contact--source-code-access)
- [License & Notes](#license--notes)

---

## Highlights ✨

- **BengFoS dataset** — gold-standard, sentence-level annotations (≈3,148 annotated sentences) for 13 FoS categories
- **Experiments** — zero-shot baselines, full & parameter-efficient fine-tuning (Adapters / LoRA), quantized deployments (16-bit), 5-fold CV, and probing analyses
- **Interpretability** — layer-wise logistic probes and token-level attention visualizations that reveal where and how FoS is encoded

---

## Dataset — BengFoS

- **Source:** Poems and lines from multiple Bengali poets (public-domain or cleared texts)
- **Labels:** 13 classes (0..12) covering None, Simile, Metaphor, Personification, Onomatopoeia, Hyperbole, Alliteration, Oxymoron/Epigram, Irony, Euphemism/Pun, Apostrophe, Synecdoche/Metonymy, Assonance
- **Splits & Preprocessing:** Stratified 5-fold CV; scripts perform Unicode normalization, tokenization, and optional augmentation/upsampling

---

## Models & Key Results

- Evaluated models include Llama-3 (8B), DeepSeek R1 Distill (7B), Mixtral (7B), and API baselines (GPT-3.5, Gemini-1.5)
- Zero-shot results are weak, motivating fine-tuning; LoRA + 16-bit quantization provides strong trade-offs for deployment

---

## Repository Layout

```text
├── README.md
├── LICENSE
├── paper/
│   └── Bengali_FoS_Identification.pdf
├── Visualizations/
│   ├── Probing Visualizations/
│   └── Deployment Visualizations/
├── data/
├── src/
├── notebooks/
├── experiments/
├── results/
└── requirements.txt
