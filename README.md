# IAC-# Internship Documentation - Dawa Seldon

Welcome to my internship documentation!  
This repository captures everything I do during my internship, including tasks, skills learned, challenges faced, and reflections.

---

## Table of Contents
- [Week 1](#week-1)
- [Week 2](#week-2) *(to be added)*
- [Week 3](#week-3) *(to be added)*
- [Week 4](#week-4) *(to be added)*

---

## Week 1: [Date Range]

**Focus of the Week:**  
The primary goal was conceptual and technical learning in AI development, specifically in **Natural Language Processing (NLP)**. The aim was to explore model architectures, fine-tuning approaches, and their limitations, rather than deploying a final product.

---

### Tasks Completed
- Conducted NLP experiments on **sentiment analysis** using movie reviews.  
- Implemented **DistilBERT** (a lightweight transformer) for supervised learning tasks:
  - Sentiment classification  
  - Text labeling  
  - Feature extraction  
- Learned that **DistilBERT cannot generate text**, due to the absence of a decoder component.  
- Began exploration of **Bhutan History chatbot (Yangtshel / BB chatbot)** using:
  - **facebook/opt-1.3b**, a large-scale causal language model capable of generating coherent text.  
- Worked with the **tokenization pipeline** to convert raw text into subword tokens for model processing.  
- Explored **Parameter-Efficient Fine-Tuning (PEFT)** using **Low-Rank Adaptation (LoRA)**:
  - Introduced small trainable adapter layers  
  - Reduced memory and compute requirements  
- Experimented with adapting the model to Bhutan History content.  
- Built a basic **interactive interface using Gradio** to test model responses.  
- Used **Google Colab** for GPU-based experimentation, learning session limitations.

---

### Skills Learned / Developed
- Understanding **transformer-based language models** and their architectures (encoder-only vs. decoder).  
- Hands-on experience with **NLP pipelines**:
- Practical knowledge of **fine-tuning limitations**, hallucinations, and the importance of dataset quality.  
- Exposure to **Parameter-Efficient Fine-Tuning (LoRA)**.  
- Gained experience with **Gradio** for rapid AI interface development.  
- Developed critical thinking about **responsible AI development**:
- Recognizing that fine-tuning alone is insufficient  
- Understanding the need to combine AI models with external knowledge sources.

---

### Challenges / Solutions
| Challenge | Solution / Learning |
|-----------|------------------|
| DistilBERT cannot generate text | Transitioned to OPT-1.3B for text generation tasks |
| Limited dataset quality for domain-specific knowledge | Noted importance of high-quality, structured datasets for deep model understanding |
| Potential model hallucinations | Learned that combining models with external knowledge (RAG) reduces inaccuracies |
| Colab session limitations | Adapted experiments to fit session constraints |

---

### Reflection / Notes
- Learned the **differences in model architectures** and how they affect task suitability.  
- Understood the importance of **dataset quality and domain adaptation** for factual accuracy.  
- Realized that **responsible AI for education** requires external knowledge integration, not only fine-tuning.  
- Gained **hands-on experience** with practical AI pipelines, tokenization, and model testing.  
- Next steps: Explore **Retrieval-Augmented Generation (RAG)** for improved factual reliability in the Bhutan History chatbot.  

---

### Suggested Images / Graphics
- **AI Hallucination Warning Graphic**: to illustrate potential inaccuracies.  
- **Knowledge + AI Integration Graphic**: to highlight combining models with external knowledge.  
- **RAG Pipeline Diagram**: for future direction in development.  

---

## Week 2: [To be added]
*(Add updates here as the week progresses)*

---

## Week 3: [To be added]
*(Add updates here as the week progresses)*

---

## Week 4: [To be added]
*(Add updates here as the week progresses)*
