# 🧠 NLP Notebook Lab

This repository contains my hands-on practice notebooks from my NLP learning journey. It includes foundational tasks and advanced workflows built using HuggingFace Transformers, LangChain, Pinecone, SBERT, and more.

---

## 📂 Topics Covered

| 📁 Category         | Description |
|---------------------|-------------|
| **Summarization**   | Using T5 model for text summarization |
| **SBERT**           | Sentence embeddings, similarity search, and clustering |
| **NER**             | Named Entity Recognition with HuggingFace |
| **Question Answering** | QA with pretrained transformer models |
| **Classification**  | Multi-class, multi-label classification using DistilBERT |
| **RAG**             | Retrieval-Augmented Generation with FAISS and Pinecone |
| **LangChain**       | Deep dives into LangChain + HuggingFace for building smart apps |
| **Pinecone**        | Vector search and indexing experiments |

---
# 🤖 Classification Mini Projects

This repository contains a collection of mini-projects focused on text classification using transformer-based models such as DistilBERT. Each subfolder demonstrates different classification tasks with hands-on training, fine-tuning, and evaluation workflows.

---

## 🗂 Project Structure

| Folder / File | Description |
|---------------|-------------|

### 📁 `SmsSpamClassifier/`
- **Goal**: Binary classification (spam vs. ham) on SMS text messages.
- **Notebook**: `sms_spam_classifier.ipynb`
- **Highlights**:
  - Basic text preprocessing
  - BOW, Count vectorization
  - NaiveBayes classifier

---

### 📁 `multiclass-distillbert-classification/`
- **Goal**: Multiclass classification of political statements as true, false, mostly-true, etc. using the **LIAR dataset**.
- **Notebooks**:
  - `truth-classifier.ipynb`
  - `truth_classifier_colab.ipynb`
  - `truth_classifier_colab_bert.ipynb`
- **Highlights**:
  - DistilBERT Model for 6-Class Classification
  - Bert Model for 6-Class Classification for higher accuracy
  - Experiments on Colab with GPU
  - Saving & reloading model
- **Folders**:
  - `models/` – saved fine-tuned models
  - `results/` – evaluation outputs
  - `data/liar_dataset/` – input data files

---

### 📁 `Multi-Label-Classification/`
- **Notebook**: `Multi_Label_Classification_Training_Demo.ipynb`
- **Goal**: Learn how to predict **multiple labels per text** (multi-label classification).
- **Highlights**:
  - Custom label binarization
  - Skill detection fine tuning with Multi-hot encoding of common skills
  - BERT adaptation for multi-label classification

---

### 📁 `Fakenews-classifier/`
- **Goal**: Classify news statements into multiple categories.
- **Notebooks**:
  - `Fakenews_Multiclass_Distillber_Finetuining.ipynb`
  - `BERT-classification-using-saved-model.ipynb`
  - `Bert_classification_training_model.ipynb`
- **Highlights**:
  - Training DistilBERT for multiclass tasks
  - Saving & loading models
  - Real-world fine-tuning example

---

## 🧠 Key Learnings

- How to fine-tune pretrained models like DistilBERT for classification tasks
- Difference between multiclass and multilabel classification
- How to organize training outputs, datasets, and models
- Running large models on Colab (GPU-compatible notebooks included)

---




## 🚀 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/subtler/NLP-Notebook-Lab.git
   cd NLP-Notebook-Lab

2.	Open any notebook (.ipynb) in Jupyter, VS Code, or Colab

3.	Follow the cells to run the code. Some notebooks may require:
	•	HuggingFace transformers
	•	Pinecone SDK
	•	GPU support (recommended for model inference)
