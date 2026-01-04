Fine-Tuned LLM for Sentiment Analysis & Contextual Responses
Project Overview

This project demonstrates an end-to-end sentiment-aware assistant for movie reviews.
It combines:
Parameter-efficient fine-tuning (LoRA) for sentiment classification
Retrieval-Augmented Generation (RAG) for contextual awareness
Generative language models for controlled and empathetic responses
Streamlit-based interactive interface

The system analyzes user movie reviews, detects sentiment (positive / negative), retrieves relevant contextual information, and generates responses grounded in retrieved context.
This project was developed as part of a hackathon.

Goals

1. Sentiment Classification
Binary classification: positive / negative
Domain-specific: movie reviews
2. Context-Aware Response Generation
Retrieval of semantically relevant examples
Response generation constrained by retrieved context
3. Efficiency
Lightweight models
CPU-friendly inference
Fast training and deployment

Dataset
IMDB Movie Reviews Dataset
Binary sentiment labels
Subsampled for efficiency:
10,000 training examples
2,000 test examples

Model Architecture
Sentiment Classifier:
Base model: distilbert-base-uncased
Fine-tuning method: LoRA (Low-Rank Adaptation)
Task: sequence classification

Why LoRA:
Trains only a small subset of parameters
Freezes base model weights
Reduces memory usage and training time

Retrieval Component
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Vector index: FAISS
Retrieves semantically similar movie review fragments

Response Generation
Generative model: distilgpt2
Prompt-based generation with explicit constraints
Responses:
2–3 sentences
Grounded in retrieved context
No introduction of external facts

System Pipeline
User Input
   ↓
Sentiment Classification (DistilBERT + LoRA)
   ↓
Context Retrieval (MiniLM + FAISS)
   ↓
Prompt Construction
   ↓
Response Generation (DistilGPT2)
   ↓
User Feedback

Training and Evaluation
Metrics:
Accuracy
Precision
Recall
F1-score
Validation loss
Training was performed for one epoch to maintain fast iteration and low computational cost.

Demo Application
Streamlit Interface
Features:
Text input for movie reviews
Sentiment detection
Context-aware response generation
Response latency measurement
User feedback collection

Technologies:
Streamlit
Hugging Face Transformers
PyTorch
ngrok

Running the Project
Install Dependencies
pip install transformers datasets peft sentence-transformers faiss-cpu streamlit pyngrok torch

Run the Streamlit App
streamlit run streamlit_app.py

Optional public access using ngrok:
from pyngrok import ngrok
ngrok.connect(8501)

Example
Input:
The movie was boring and too long.
Detected sentiment: Negative
Generated response:
The reviewer felt the movie was too long, and the plot failed to keep the audience engaged. This aligns with your impression that the film lacked enough interest to sustain its length.

Feedback Loop

Users can rate generated responses on a 1–5 scale.
Feedback is stored in feedback.csv and can be used for future improvements.

Future Improvements
Replace rule-based sentiment detection in the UI with the fine-tuned classifier
Improve retrieval with a larger and more diverse corpus
Add multilingual support
Train a unified instruction-tuned model
Introduce response explainability

Hackathon Context
This project was developed as a hackathon solution focusing on:
Efficient LLM fine-tuning
Retrieval-augmented generation
Practical and deployable NLP systems
