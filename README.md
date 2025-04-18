# 🚀 Offensive Language Detection Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

A complete and flexible pipeline for fine-tuning and evaluating transformer-based models on the **Offensive Language Identification task**. Supports **English subtasks A, B, and C**, as well as **Danish subtask A**. Built on the OLID dataset (English) and the Danish offensive language dataset.

---

## 📑 Table of Contents

- [📄 Project Overview](#project-overview)
- [⚙️ Setup](#setup)
- [🗂️ Repository Structure](#repository-structure)
- [📌 Code Architecture](#code-architecture)
  - [🔹 `training.py` – Fine-Tuning and Evaluating Models from scratch](#trainingpy--fine-tuning-models)
  - [🔹 `evaluation.py` – Evaluation of our four fine-tuned models, identified as the best-performing through extensive experimentation](#evaluationpy--model-evaluation)
- [🌟 Features](#features)
- [🏷️ Branching Strategy](#branching-strategy)
- [📝 Citation](#citation)

---

## 📄 Project Overview

This repository supports fine-tuning and evaluation for the following tasks:

| Subtask | Description |
|:-:|:-|
| **Task A (English)** | Offensive vs. Non-Offensive classification |
| **Task B (English)** | Targeted vs. Untargeted offense classification |
| **Task C (English)** | Target Type: Individual, Group, Other |
| **Task A (Danish)** | Offensive vs. Non-Offensive classification |

---

## ⚙️ Setup

Before running any scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Repository Structure

```
├── data/
├── src/
│   ├── training.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   └── utils.py
├── .gitlab-ci.yml
├── LICENSE
├── README.md
└── requirements.txt            
```

---

## 📌 Code Architecture

The project consists of two main scripts, which have to be run from the src folder directly:

### 🔹 `training.py` – Fine-Tuning Models

This script handles **model training** for all supported tasks. Then you can customize:

- **The subtask:** Task A/B/C in English or Task A in Danish
- **Model architecture:** e.g., `bert-base-cased`, `albert-base-v2`, `bert-base-multilingual-cased`
- **Preprocessing methods:** If no method is specified, the default is no preprocessing
- **Dataset size:** Default is a sample of **1000 tweets**; you can train on the **full dataset**
- **Hyperparameters:** Batch size, learning rate, epochs  
  *(Default settings: `batch size = 8`, `learning rate = 2e-5`, `epochs = 3`)*  
  **Note:** Only one hyperparameter may be changed at a time while keeping others at default.

After training, the model is **saved and evaluated** on the test set automatically.

**Usage Example:**
```bash
python training.py --task A-eng --model albert-base-v2 --preprocessing-data lowercasing --hyperparameters batch-16 --full-dataset
```
**Note:** (When in src folder, otherwise add "src/" before the training.py)

#### Command-Line Options

| Argument | Description | Choices / Examples |
|:-:|:-|:-|
| `-h, --help` | Show help message and exit | |
| `--task` | Choose the task to fine-tune the model on | `{A-eng, B-eng, C-eng, A-danish}` |
| `--model` | Specify the model architecture | `{bert-base-cased, albert-base-v2, bert-base-multilingual-cased}` |
| `--hyperparameters` | Set one hyperparameter at a time | `{batch-1, batch-16, epochs-6, lr-1e-5, lr-3e-5}`<br>*(Default: batch_size=8, epochs=3, lr=2e-5)* |
| `--full-dataset` | Train on the full dataset instead of default subset | *(Flag, no value required)* |
| `--preprocessing-data` | Specify one or more preprocessing techniques | `{remove-hashtags-urls-at, lowercasing, expand-emojis, remove-stopwords, stemming}` |

---

### 🔹 `evaluation.py` – Model Evaluation

This script loads our four fine-tuned models - identified as the best-performing through extensive experimentation. The models are hosted on Hugging Face and are automatically retrieved for evaluation. The script supports all four subtasks and reports the following after evaluation:

- **Macro F1-Score** (Primary metric)
- **Accuracy**

**Usage Example:**
```bash
python evaluation.py --model bert-base-cased-task-A-eng
```
**Note:** (When in src folder, otherwise add "src/" before the evaluation.py)

#### Command-Line Options

| Argument | Description | Choices |
|:-:|:-|:-|
| `-h, --help` | Show help message and exit | |
| `--model` | Choose one of the pre-trained models to evaluate | `{bert-base-cased-task-A-eng, bert-base-cased-task-B-eng, bert-base-cased-task-C-eng, bert-base-multilingual-cased-task-A-danish}` |

---

## 🌟 Features

✅ Fine-tuning & Evaluation for **4 subtasks**  
✅ Multiple **Transformer architectures** supported  
✅ Configurable **Preprocessing pipeline**  
✅ Flexible **Hyperparameter tuning**  
✅ Supports **English & Danish datasets**  
✅ Evaluation metrics: **Macro F1-Score & Accuracy**  
✅ Best fine-tuned models available on Hugging Face

---

## 🏷️ Branching Strategy

The repository uses the following branching structure:

| Branch | Description |
|:-:|:-|
| `main` | Contains the **final code** and four **best fine-tuned models** |
| `add-bert-model` | Experimental results - BERT components |
| `albert-model` | Experimental results - ALBERT models |
| `roberta-model` | Experimental results - RoBERTa models |

---

## 📝 Citation

If you use this repository for your research or coursework, please consider citing:

> Zampieri et al. (2020). SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). In *Proceedings of the Fourteenth Workshop on Semantic Evaluation (SemEval-2020)*.

---
