# 📚 SlideScholar

**A Multimodal RAG-Based AI Study Assistant for Lecture Slides**

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-blue)](https://huggingface.co/spaces/kshamaasuresh/SlideScholar)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Try it now →** [huggingface.co/spaces/kshamaasuresh/SlideScholar](https://huggingface.co/spaces/kshamaasuresh/SlideScholar)

SlideScholar is a zero-cost, open-source AI study assistant that ingests PDF and PPTX lecture slides, understands both text and visual content through a multimodal pipeline, and generates grounded study materials — all sourced exclusively from the student's own slides.

Built as the final project for **STATGR5293: Generative AI Using LLMs** (Spring 2026) at Columbia University.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Ingestion Pipeline (Notebook 1)](#running-the-ingestion-pipeline-notebook-1)
  - [Running RAG & Evaluation (Notebook 2)](#running-rag--evaluation-notebook-2)
  - [Running the Gradio UI Locally](#running-the-gradio-ui-locally)
- [Reproducing Results](#reproducing-results)
- [Output Modes](#output-modes)
- [Evaluation Results](#evaluation-results)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## Key Features

- **Multimodal Slide Understanding** — LLaVA-7B processes every slide as an image, capturing diagrams, equations, and charts that text-only parsers silently discard.
- **Slide-Aware Semantic Chunking** — Each slide is one indivisible semantic unit (merged raw text + vision description) with metadata for faithful source citation.
- **Cross-Lecture Retrieval** — FAISS-powered semantic search across all uploaded lectures, connecting concepts from different weeks.
- **5 Grounded Output Modes** — Study Guide, Flashcards, Practice Exam, ELI5, and Gap Analysis — all with `[Source N]` slide citations.
- **Adaptive Gap Detection Loop** — Wrong practice exam answers trigger re-retrieval of relevant source slides and targeted re-explanation, implementing evidence-based retrieval practice.
- **Zero Cost** — All models are open-weight. Runs on Google Colab (A100) for development and HuggingFace Spaces (free tier) for deployment.

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   INGEST    │────▶│  UNDERSTAND  │────▶│    INDEX    │────▶│   RETRIEVE   │────▶│   GENERATE   │
│             │     │              │     │             │     │              │     │              │
│ pdfplumber  │     │  LLaVA-7B    │     │ e5-large-v2 │     │  Top-k FAISS │     │ Mistral-7B   │
│ python-pptx │     │  4-bit quant │     │ FAISS       │     │  cosine sim  │     │ 4-bit NF4    │
│ pdf2image   │     │              │     │ 1024-dim    │     │              │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘     └──────────────┘
```

**Pipeline flow:**

1. **Ingest** — `pdfplumber` and `python-pptx` extract raw text; `pdf2image` renders each slide as a PNG.
2. **Understand** — LLaVA-7B (4-bit quantized via `bitsandbytes`) generates a natural-language visual description per slide, capturing diagrams, equations, and charts.
3. **Index** — Merged text+vision chunks are embedded with `intfloat/e5-large-v2` (1024-dim) and stored in a FAISS `IndexFlatIP` index.
4. **Retrieve** — At query time, top-k cosine-similar chunks are fetched with full slide metadata.
5. **Generate** — Mistral-7B-Instruct (4-bit NF4) produces study materials strictly grounded in retrieved context.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Slide Parsing | `pdfplumber`, `python-pptx`, `pdf2image` | Text extraction + slide PNG rendering |
| Vision Model | LLaVA-7B (`llava-hf/llava-1.5-7b-hf`) | Visual understanding of diagrams, equations, charts |
| Embeddings | `intfloat/e5-large-v2` (1024-dim) | Dense vector embeddings for semantic search |
| Vector Store | FAISS `IndexFlatIP` | Cosine similarity retrieval |
| Generation LLM | `mistralai/Mistral-7B-Instruct-v0.2` (4-bit NF4) | Grounded study material generation |
| Evaluation | RAGAS + custom metrics | Faithfulness, relevance, context recall |
| UI Framework | Gradio (6 tabs) | Web interface with file upload |
| Deployment | HuggingFace Spaces (CPU + Inference API) | Public demo at zero cost |
| Dev Environment | Google Colab (A100 GPU) | Model inference, batch processing |

---

## Data

1. Sample lecture slides used for evaluation are available here:
[Download Slides (Google Drive)](https://drive.google.com/drive/folders/16vd7KXSQ2RpodFWF2Rqq_ET0Gb_jJySD)
2. Demo Video available here:
[Demo Video (Google Drive)](https://drive.google.com/file/d/1SPgRckxhk70muZiz1z-lbLS1DsriaF_W/view?usp=drive_link)

---

## Repository Structure

```
SlideScholar/
├── README.md
├── app.py                                          # Gradio UI — HuggingFace Spaces entry point
├── requirements.txt                                # Pinned dependencies for Spaces
├── chunks.json                                     # Pre-built chunk index (from Notebook 1)
├── slidescholar.faiss                              # FAISS index (from Notebook 2)
├── notebooks/
│   ├── SlideScholar_01_Ingestion.ipynb             # Notebook 1: PDF/PPTX → chunks.json
│   └── SlideScholar_02_RAG_Generation_Evaluation.ipynb  # Notebook 2: Embeddings, RAG, eval
└── docs/
    └── SlideScholar_Final_Presentation.pptx        # Project presentation slides
```

---

## Getting Started

### Prerequisites

- Google account with access to [Google Colab](https://colab.research.google.com/)
- Colab runtime set to **GPU → A100** (required for LLaVA-7B)
- A [HuggingFace account](https://huggingface.co/) with an API token (for gated model access to Mistral-7B)
- Store your HF token in Colab: **Secrets** (key icon in left sidebar) → add `HF_TOKEN`

### Running the Ingestion Pipeline (Notebook 1)

1. Open `notebooks/SlideScholar_01_Ingestion.ipynb` in Google Colab.
2. Set runtime to **A100 GPU**.
3. Mount Google Drive — the notebook writes to `MyDrive/SlideScholar/`.
4. Place your lecture PDFs/PPTXs in `MyDrive/SlideScholar/Slides_Input/`.
5. Run all cells. The notebook will:
   - Parse PDFs and PPTXs (text extraction + slide PNG rendering)
   - Run LLaVA-7B vision descriptions on every slide
   - Build chunk dictionaries with metadata
   - Save `chunks.json` to Google Drive

**Output:** `chunks.json` — the only file Notebook 2 needs.

### Running RAG & Evaluation (Notebook 2)

1. Open `notebooks/SlideScholar_02_RAG_Generation_Evaluation.ipynb` in Google Colab.
2. Set runtime to **A100 GPU**.
3. Ensure `HF_TOKEN` is set in Colab Secrets.
4. Run all cells. The notebook will:
   - Load `chunks.json` from Drive
   - Build the FAISS index with `e5-large-v2` embeddings
   - Load Mistral-7B-Instruct (4-bit quantized)
   - Test all 5 generation modes
   - Run RAGAS evaluation on 50 grounded QA pairs
   - Save `slidescholar.faiss` to Google Drive

**Output:** `slidescholar.faiss` + evaluation results.

### Running the Gradio UI Locally

```bash
# Clone the repo
git clone https://github.com/KshamaaS/SlideScholar.git
cd SlideScholar

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# Ensure chunks.json and slidescholar.faiss are in the repo root

# Launch
python app.py
```

The app will be available at `http://localhost:7860`.

Or use the hosted version: **[huggingface.co/spaces/kshamaasuresh/SlideScholar](https://huggingface.co/spaces/kshamaasuresh/SlideScholar)**

---

## Reproducing Results

This section provides a complete guide to reproduce the SlideScholar pipeline end-to-end, from raw lecture slides to a live deployed application.

### Environment

| Requirement | Specification |
|---|---|
| Python | 3.10+ |
| GPU (Notebooks) | NVIDIA A100 (Google Colab Pro) — required for LLaVA-7B and Mistral-7B |
| GPU (Deployment) | None — HuggingFace Spaces CPU basic tier with Inference API |
| Disk (Colab) | ~15 GB for model weights (cached after first download) |
| HuggingFace Token | Required — Mistral-7B-Instruct is a gated model |

### Step-by-Step Reproduction

**Step 1 — Clone the repository**
```bash
git clone https://github.com/KshamaaS/SlideScholar.git
cd SlideScholar
```

**Step 2 — Prepare lecture slides**

Place PDF and/or PPTX lecture files into a folder. The ingestion pipeline expects them in Google Drive at `MyDrive/SlideScholar/Slides_Input/`. Sample test slides from our course (STATGR5293) were used for evaluation — any English-language lecture slides will work.

**Step 3 — Run Notebook 1 (Ingestion)**

Open `notebooks/SlideScholar_01_Ingestion.ipynb` in Google Colab.

1. Set runtime: **Runtime → Change runtime type → A100 GPU**
2. Add your HuggingFace token: **Secrets** (key icon) → add key `HF_TOKEN` with your token
3. Mount Google Drive (cell runs automatically)
4. Run all cells sequentially

The notebook will:
- Install all dependencies (pinned versions)
- Parse PDFs with `pdfplumber` + render slide PNGs with `pdf2image`
- Parse PPTXs with `python-pptx` + render via LibreOffice
- Run LLaVA-7B (4-bit quantized) vision descriptions on each slide
- Build chunk dictionaries with metadata (lecture name, slide number, week, chunk ID)
- Save `chunks.json` to Google Drive
- Run verification checks on output quality

**Expected output:** `MyDrive/SlideScholar/chunks.json`
**Expected runtime:** ~2–4 minutes per lecture deck (A100), depending on slide count.

**Step 4 — Run Notebook 2 (Embeddings, RAG, Generation & Evaluation)**

Open `notebooks/SlideScholar_02_RAG_Generation_Evaluation.ipynb` in Google Colab.

1. Same runtime setup as Notebook 1 (A100 + HF_TOKEN)
2. Run all cells sequentially

The notebook will:
- Load `chunks.json` from Drive
- Initialize `intfloat/e5-large-v2` and embed all chunks (batched, GPU-accelerated)
- Build and save the FAISS `IndexFlatIP` index
- Load `mistralai/Mistral-7B-Instruct-v0.2` (4-bit NF4 quantization)
- Test all 5 generation modes (study guide, flashcards, exam, ELI5, gap analysis)
- Run RAGAS evaluation on 50 grounded QA pairs
- Print evaluation metrics summary

**Expected outputs:** `MyDrive/SlideScholar/slidescholar.faiss` + evaluation metrics
**Expected runtime:** ~10–15 minutes total (A100).

**Step 5 — Deploy to HuggingFace Spaces**

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio, Hardware: CPU basic)
2. Upload these files to the Space repo:
   - `app.py`
   - `requirements.txt`
   - `chunks.json` (from Step 3)
   - `slidescholar.faiss` (from Step 4)
3. Add your HuggingFace token as a Space Secret: **Settings → Secrets → `HF_TOKEN`**
4. The Space will build and launch automatically at your public URL

**Alternative — run locally:**
```bash
pip install -r requirements.txt
export HF_TOKEN="hf_your_token_here"
python app.py
# Open http://localhost:7860
```

### Dependency Versions

All dependencies are pinned in `requirements.txt` and in the notebook install cells to ensure reproducibility. Key versions:

| Package | Version |
|---|---|
| `torch` | ≥ 2.1.0 |
| `transformers` | 4.40.0 |
| `sentence-transformers` | 2.7.0 |
| `faiss-cpu` | 1.8.0 |
| `bitsandbytes` | 0.45.3 |
| `pdfplumber` | 0.11.0 |
| `gradio` | ≥ 4.28.3 |
| `Pillow` | 9.5.0 |
| `python-pptx` | 0.6.23 |

### Verification

After running both notebooks, run the verification cell at the end of Notebook 1 (`verify_chunks()`) and the evaluation cell at the end of Notebook 2 to confirm:
- All chunks have valid structure and non-empty text
- FAISS index vector count matches chunk count
- Evaluation metrics are within expected ranges

---

## Output Modes

| Mode | Description | Output Format |
|---|---|---|
| 📝 **Study Guide** | Structured notes with key concepts, definitions, formulas | Markdown with `[Source N]` citations |
| 🃏 **Flashcards** | 10 Q&A pairs with source tracing | JSON array, rendered as cards |
| 📋 **Practice Exam** | 5 MCQs + 3 short-answer questions, graded difficulty | Formatted exam with answer key |
| 💡 **ELI5** | Plain-language explanation with real-world analogies | Conversational markdown |
| 🎯 **Gap Analysis** | Re-explanation of wrong answers using source slides | Why correct, why wrong, memory aid |

All outputs are grounded exclusively in the student's uploaded lecture slides — not generic AI knowledge.

---

## Evaluation Results

Evaluation was performed on a test set of 50 grounded QA pairs derived from actual course slides.

| Metric | Score | Target | Status |
|---|---|---|---|
| Hit Rate | 0.80 | ≥ 0.80 | ✅ Met |
| Retrieval F1 | 0.75 | ≥ 0.70 | ✅ Met |
| Answer Relevance | 0.78 | ≥ 0.75 | ✅ Met |
| Faithfulness | 0.72 | ≥ 0.80 | ⚠️ Partial |

**Faithfulness gap analysis:** When retrieved context is sparse on a topic, Mistral-7B supplements with parametric knowledge. This is expected RAG behavior — faithfulness improves as more lectures are added to the corpus.

**Baseline comparison:** ChatGPT without access to slides scores 0.60 on answer relevance and 0.35 on faithfulness, demonstrating the value of grounded retrieval.
---

## Team

| Member | UNI | Branch |
|---|---|---|
| **Chenglu Xing** | cx2353 | `env_l` |
| **Kshamaa Suresh** | ks4423 | `env_k` |
| **Simryn Parikh** | sap2254 | `env_s` |

**Git workflow:** Personal branches (`env_k`, `env_l`, `env_s`) → `dev` (integration) → `main` (release). Branch protection enabled on both `dev` and `main`.

---

## Acknowledgments

- **Instructor:** Chen Wang (cw3687)
- **TA:** Christopher Ng (cn2673)
- **Course:** STATGR5293 — Generative AI Using LLMs, Spring 2026, Columbia University

Built with [LLaVA](https://llava-vl.github.io/), [Mistral AI](https://mistral.ai/), [FAISS](https://github.com/facebookresearch/faiss), [Gradio](https://gradio.app/), and [HuggingFace](https://huggingface.co/).
