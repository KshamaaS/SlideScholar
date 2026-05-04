---
title: SlideScholar
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
hardware: cpu-basic
---

# 📚 SlideScholar

A multimodal RAG-based AI study assistant that generates study guides, flashcards,
practice exams, and ELI5 explanations grounded in your actual lecture slides.

## How it works

1. Lecture slides are ingested and embedded using `intfloat/e5-large-v2`
2. Student queries are matched against slides using FAISS semantic search
3. Retrieved slides are passed to `Mistral-7B-Instruct` via the HuggingFace Inference API
4. All outputs cite the specific slides they came from

## Stack

| Component | Details |
|---|---|
| Embeddings | `intfloat/e5-large-v2` (local, CPU) |
| Vector Store | FAISS IndexFlatIP (local) |
| Generation | `mistralai/Mistral-7B-Instruct-v0.2` (HF Inference API) |
| Hardware | CPU basic (free tier) |

## Features

- 📝 **Study Guide** — structured notes with slide citations
- 🃏 **Flashcards** — 10 Q&A pairs per topic
- 📋 **Practice Exam** — 5 MCQ + 3 short answer with answer key
- 💡 **ELI5** — simple explanation with analogy
- 🎯 **Gap Analysis** — wrong answer → targeted re-explanation from slides
- 📂 **Upload Slides** — add new PDFs to the index on the fly

## Course

STATGR5293 · Generative AI Using LLMs · Spring 2026 · Columbia University