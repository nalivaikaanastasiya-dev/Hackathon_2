# Hackathon_2

Sentiment-Aware Assistant with LoRA and Retrieval-Augmented Generation:

Project Overview:
This project implements a sentiment-aware conversational assistant based on a parameter-efficient fine-tuned Large Language Model (LLM).
The system performs sentiment classification and generates context-aware responses by combining LoRA fine-tuning with retrieval-augmented generation (RAG).
The solution is designed to be lightweight, efficient, and runnable on limited hardware, making it suitable for hackathons and rapid prototyping.

Objectives:
Fine-tune a transformer model for sentiment classification using LoRA
Build a semantic retrieval system using dense embeddings and FAISS
Generate context-aware responses by augmenting prompts with retrieved documents
Deploy an interactive Streamlit UI
Evaluate the system end-to-end (accuracy, generation quality, latency)
(Bonus) Collect human feedback for future fine-tuning

System Architecture:
User Input
   │
   ▼
Sentiment Classifier (LoRA Fine-Tuned DistilBERT)
   │
   ├── Sentiment Label
   ▼
Retriever (Sentence-Transformers + FAISS)
   │
   ├── Top-k Relevant Documents
   ▼
Prompt Builder (User Input + Sentiment + Context)
   │
   ▼
Response Generator (Causal LLM)
   │
   ▼
Streamlit UI Output

Technologies Used:
| Module       | Tools / Libraries              |
| ------------ | ------------------------------ |
| Base Model   | distilbert-base-uncased        |
| Fine-Tuning  | PEFT (LoRA)                    |
| Data         | Hugging Face Datasets          |
| Tokenization | Transformers                   |
| Retrieval    | FAISS                          |
| Embeddings   | sentence-transformers          |
| Generation   | Hugging Face `generate()`      |
| UI           | Streamlit                      |
| Evaluation   | Accuracy, BLEU, ROUGE, Latency |

Dataset:
IMDB Movie Reviews
Binary sentiment labels: positive / negative
Subsampled to 10,000 examples for efficiency

1. Setup Instructions:
git clone https://github.com/nalivaikaanastasiya-dev/sentiment-aware-assistant.git
cd sentiment-aware-assistant
2. Install Dependencies
pip install -r requirements.txt

Main dependencies:
transformers
datasets
peft
sentence-transformers
faiss-cpu
streamlit
torch

Training the Sentiment Classifier:
The sentiment classifier is trained using LoRA, which freezes the base model and trains only low-rank adapters.
Key training settings:
Learning rate: 1e-4
Batch size: 8
Epochs: 3
Mixed precision (fp16) if available
python train_classifier.py

Retrieval Component:
Dense embeddings generated using all-MiniLM-L6-v2
Indexed with FAISS
Retrieves top-k relevant documents for each user query
This retrieved context is injected into the generation prompt.

Context-Aware Response Generation
The system constructs prompts using:
User input
Predicted sentiment
Retrieved contextual documents
Example prompt structure:
User input: ...
Detected sentiment: ...
Relevant context: ...
Generate a helpful and empathetic response.

Running the Streamlit App
streamlit run app.py
Features:
Text input from user
Sentiment prediction display
Context-aware generated response
Optional feedback rating (bonus)

Evaluation:
The system is evaluated end-to-end using:
Sentiment Accuracy
BLEU / ROUGE scores for generated responses
Latency (response time per query)
Testing was performed on 50 real and synthetic queries.

Human Feedback Loop (Bonus)
The UI includes a feedback mechanism where users can rate generated responses.
Feedback is stored locally (CSV)
Can be reused for:
Offl
Incremental fine-tuning
Reinforcement learning experiments

Optimization & Efficiency:
LoRA reduces trainable parameters by >95%
Suitable for CPU-only or single-GPU environments
Subsampled dataset for faster experimentation
Mixed precision for speed-up where supported

Limitations & Future Work:
Generation model is not fine-tuned (can be improved)
Retrieval corpus is limited to training data
Future improvements:
Online learning from feedback
Multi-class sentiment
Larger retrieval corpus
Better prompt engineering

Conclusion:
This project demonstrates how parameter-efficient fine-tuning and retrieval-augmented generation can be combined to build a powerful yet efficient sentiment-aware assistant.
It is well-suited for hackathons, educational purposes, and rapid NLP prototyping

Author:
Nalivaika Anastssiya
