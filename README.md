**Fine-Tuned LLM for Sentiment Analysis & Contextual Responses**

**Overview**

This project presents a sentiment-aware assistant for movie reviews that combines efficient fine-tuning, retrieval-augmented generation, and controlled text generation.

The system:

* Classifies review sentiment (positive / negative)
* Retrieves relevant contextual information
* Generates empathetic, context-grounded responses

**Key Features:**
LoRA fine-tuning of distilbert-base-uncased for sentiment classification
Retrieval-Augmented Generation (RAG) using MiniLM embeddings + FAISS
Controlled response generation with distilgpt2
Lightweight and CPU-friendly pipeline
Interactive Streamlit demo with feedback collection

**Architecture**
```
User Input
   ↓
Sentiment Classifier (DistilBERT + LoRA)
   ↓
Context Retrieval (MiniLM + FAISS)
   ↓
Prompt Construction
   ↓
Response Generation (DistilGPT2)
```
**Dataset**
IMDB Movie Reviews Dataset
Binary sentiment labels
Subsampled:
10,000 training examples
2,000 test examplesModels

**Models**
Sentiment Classification: distilbert-base-uncased + LoRA
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Generation: distilgpt2

**Training & Evaluation**
Metrics: Accuracy, Precision, Recall, F1-score
Training: 1 epoch (fast iteration, low compute cost)

**Demo**
Built with Streamlit
Features:
Review input
Sentiment detection
Context-aware response generation
Latency measurement
User feedback (CSV)
Run locally:
```
streamlit run streamlit_app.py
```

**Example**
Input:
The movie was boring and too long.
Output:
Sentiment: Negative
Response:
The reviewer felt the movie was too long, and the plot failed to keep the audience engaged.

**Why This Matters**
Demonstrates practical LLM deployment
Shows efficient fine-tuning without full retraining
Reduces hallucinations via context grounding
Easily extensible to other domains

**Future Work**
Replace rule-based UI sentiment with the fine-tuned classifier
Expand retrieval corpus
Add multilingual support
Train a unified instruction-tuned model
