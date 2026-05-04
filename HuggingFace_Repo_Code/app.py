"""
SlideScholar — app.py
HuggingFace Spaces deployment (CPU basic — free tier).
Generation via HuggingFace Inference API — no local GPU required.
Files required at repo root: chunks.json  slidescholar.faiss
"""

import os, re, json
from pathlib import Path
from typing  import List, Dict, Tuple, Optional

import numpy as np
import faiss
import torch

import gradio_client.utils as _gcu
_orig = _gcu.json_schema_to_python_type
def _safe(schema, defs=None):
    if not isinstance(schema, dict): return "Any"
    try: return _orig(schema)
    except Exception: return "Any"
_gcu.json_schema_to_python_type = _safe

import gradio as gr
from sentence_transformers import SentenceTransformer

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
CHUNKS_PATH = Path("chunks.json")
FAISS_PATH  = Path("slidescholar.faiss")
MISTRAL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"
_db         = None
_PLACEHOLDER = "SSANSWER"


# ══════════════════════════════════════════════════════════════════════════════
# VDB CLASS
# ══════════════════════════════════════════════════════════════════════════════

class vdb:
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = SentenceTransformer(model_name, device=self.device)
        self.dim    = 1024
        self.index  = None
        self.chunks: List[Dict] = []

    def _embed(self, texts, prefix):
        return self.model.encode(
            [f"{prefix}{t}" for t in texts],
            batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def load(self, faiss_path, chunks):
        self.index  = faiss.read_index(faiss_path)
        self.chunks = chunks

    def search(self, query, top_k=8):
        if self.index is None: raise RuntimeError("Index not loaded.")
        q = self._embed([query], "query: ")
        distances, indices = self.index.search(q, top_k)
        return [
            {"score": float(d), "chunk": self.chunks[i]}
            for d, i in zip(distances[0], indices[0])
            if i != -1 and i < len(self.chunks)
        ]

    def add_texts(self, texts, metadatas):
        self.chunks.extend([{"text": t, "metadata": m} for t, m in zip(texts, metadatas)])
        emb = self._embed(texts, "passage: ")
        if self.index is None: self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)


def _load_index():
    global _db
    if _db is not None: return _db
    if not CHUNKS_PATH.exists(): raise FileNotFoundError("chunks.json not found in repo root.")
    if not FAISS_PATH.exists():  raise FileNotFoundError("slidescholar.faiss not found in repo root.")
    with open(CHUNKS_PATH) as f: chunks = json.load(f)
    _db = vdb()
    _db.load(str(FAISS_PATH), chunks)
    print(f"Index loaded — {_db.index.ntotal} vectors")
    return _db


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _generate(prompt: str, max_new_tokens: int = 700, temperature: float = 0.1) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN secret not set in Space Settings → Secrets.")
    try:
        from huggingface_hub import InferenceClient
        client   = InferenceClient(model=MISTRAL_ID, token=HF_TOKEN)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Inference API error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_context(results):
    parts = []
    for i, r in enumerate(results):
        m = r["chunk"]["metadata"]
        parts.append(
            f"[Source {i+1}: {m.get('name','?')} — Slide {m.get('slide','?')}]\n"
            f"{r['chunk']['text'][:900]}"
        )
    return "\n\n" + ("\n\n" + "─"*40 + "\n\n").join(parts)

def _fmt_sources(results):
    lines = ["---", "**📚 Sources used:**"]
    for r in results:
        m = r["chunk"]["metadata"]
        lines.append(f"- **{m.get('name','?')}** — Slide {m.get('slide','?')} *(score: {r['score']:.2f})*")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EXAM FORMATTER
# Handles all model output styles: **A)**, [A], (A), plain A)
# Tested against exact screenshot output patterns.
# ══════════════════════════════════════════════════════════════════════════════

def _fix_exam_format(text: str) -> str:
    """
    Post-process exam output to put each MCQ option on its own line.

    Strategy:
      1. Strip ALL ** markers — eliminates **A)**, **Answer: B)** ambiguity
      2. Normalize [A] and (A) styles → plain A)
      3. Replace answer markers with a placeholder (so D) in Answer: D) isn't split)
      4. Split every option onto its own line
      5. Re-bold options at line-start
      6. Restore answer placeholder with clean ✅ formatting
    """
    # 1. Strip ALL bold markers — re-add selectively in steps 5 & 6
    text = text.replace('**', '')

    # 2. Normalize remaining option styles → plain "X) "
    text = re.sub(r'\[([ABCD])\]\s*',    r'\1) ', text)   # [A]  → A)
    text = re.sub(r'\(([ABCD])\)\s*',     r'\1) ', text)   # (A)  → A)
    text = re.sub(r'([ABCD])\)([^\s)])', r'\1) \2', text)  # A)x  → A) x

    # 3. Replace answer markers with placeholder BEFORE splitting options
    #    Matches: optional ✅, optional [, "Answer", optional :, letter, optional ]/)
    text = re.sub(
        r'[✅✓]?\s*\[?Answer:?\s*([ABCD])[\])]?\s*',
        lambda m: f' {_PLACEHOLDER}_{m.group(1)} ',
        text
    )

    # 4. Put each option on its own line
    #    Matches any non-newline char + spaces + option marker
    text = re.sub(r'([^\n]) *([ABCD]\) )', r'\1\n\2', text)

    # 5. Bold options that are now at the start of a line
    text = re.sub(r'^([ABCD])\) ', r'**\1)** ', text, flags=re.MULTILINE)

    # 6. Restore answer placeholder → clean formatted answer line
    text = re.sub(rf'\s*{_PLACEHOLDER}_([ABCD])\s*', r'\n\n✅ **Answer: \1)** ', text)

    # 7. Clean up whitespace
    text = re.sub(r' {2,}',  ' ',    text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def _study_prompt(context, query):
    return f"""You are an AI tutor helping students study for exams.
STRICT RULES:
1. Use ONLY the lecture slide content below. Do not add outside knowledge.
2. Copy key terms, formulas, and definitions verbatim from the slides.
3. Cite every point with [Source N].
4. If something is not covered, say "Not covered in provided slides."

Context:
{context}

Question: {query}

Output (use these exact headers):
## Key Concepts
## Definitions
## Important Formulas
## Common Exam Topics
## Summary"""


def _flashcard_prompt(context, query):
    return f"""Generate 10 flashcard Q&A pairs from these lecture slides.

Return ONLY a JSON array. Start immediately with [ and end with ].
No explanation. No markdown fences. Just the JSON.

Each item: {{"q": "question", "a": "answer", "source": "Lecture X Slide Y"}}

Context:
{context}

Topic: {query}

["""


def _exam_prompt(context, query):
    return f"""Create a practice exam from these lecture slides only.
Use ONLY information from the provided slides.

## Practice Exam: {query}

---

### Multiple Choice (5 questions)

**Question 1.** [Easy] Question text here?
A) First option
B) Second option
C) Third option
D) Fourth option
✅ **Answer: B)** Short explanation. [Source N]

---

**Question 2.** [Medium] Question text here?
A) First option
B) Second option
C) Third option
D) Fourth option
✅ **Answer: A)** Short explanation. [Source N]

---

**Question 3.** [Medium] Question?
A) ...
B) ...
C) ...
D) ...
✅ **Answer: ?)** Explanation. [Source N]

---

**Question 4.** [Hard] Question?
A) ...
B) ...
C) ...
D) ...
✅ **Answer: ?)** Explanation. [Source N]

---

**Question 5.** [Hard] Question?
A) ...
B) ...
C) ...
D) ...
✅ **Answer: ?)** Explanation. [Source N]

---

### Short Answer (3 questions)

**Question 6.** [Medium] Question?
**Model Answer:** Full answer. [Source N]

---

**Question 7.** [Hard] Question?
**Model Answer:** Full answer. [Source N]

---

**Question 8.** [Hard] Question?
**Model Answer:** Full answer. [Source N]

---

### Answer Key
1-?  2-?  3-?  4-?  5-?

---

Lecture slides:
{context}

Topic: {query}

Begin:"""


def _eli5_prompt(context, query):
    return f"""Explain this concept to someone who has never heard of it.
Use ONLY the provided lecture slides. Write in plain friendly language.

## Simple Explanation

**The core idea in one sentence:**
Write one clear sentence here.

**Real-world analogy:**
Compare it to something familiar from everyday life.

**How it works, step by step:**
1. First step
2. Second step
3. Third step

**Why it matters:**
One paragraph on practical importance.

Context:
{context}

Concept: {query}

Explanation:"""


# ══════════════════════════════════════════════════════════════════════════════
# FLASHCARD PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_flashcards(raw: str) -> List[Dict]:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    if not cleaned.startswith("["): cleaned = "[" + cleaned
    last = cleaned.rfind("}")
    if last != -1 and not cleaned.rstrip().endswith("]"):
        cleaned = cleaned[:last + 1] + "]"
    try:
        result = json.loads(cleaned)
        if isinstance(result, list) and result: return result
    except json.JSONDecodeError:
        pass
    cards = []
    for m in re.finditer(
        r'"q"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"a"\s*:\s*"((?:[^"\\]|\\.)*)"'
        r'(?:\s*,\s*"source"\s*:\s*"((?:[^"\\]|\\.)*)")?',
        cleaned, re.DOTALL
    ):
        cards.append({
            "q":      m.group(1).replace('\\"', '"'),
            "a":      m.group(2).replace('\\"', '"'),
            "source": (m.group(3) or "").replace('\\"', '"'),
        })
    if not cards: raise ValueError("No flashcards found in output")
    return cards


def _render_flashcards(cards, query):
    lines = [f"## 🃏 Flashcards: *{query}*", f"*{len(cards)} cards generated*", ""]
    for i, card in enumerate(cards, 1):
        q = card.get("q", "").strip()
        a = card.get("a", "").strip()
        s = card.get("source", "").strip()
        lines += ["---", f"**Q{i}.** {q}", "", f"> {a}"]
        if s: lines += ["> ", f"> *📖 {s}*"]
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TAB HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def handle_study_guide(query, k):
    if not query.strip(): return "Please enter a topic or question.", ""
    try:
        db = _load_index()
        results = db.search(query, top_k=int(k))
        output  = _generate(_study_prompt(_fmt_context(results), query), max_new_tokens=900)
        return output, _fmt_sources(results)
    except Exception as e:
        return f"❌ {e}", ""


def handle_flashcards(query, k):
    if not query.strip(): return "Please enter a topic.", ""
    try:
        db = _load_index()
        results = db.search(query, top_k=int(k))
        raw     = _generate(_flashcard_prompt(_fmt_context(results), query), max_new_tokens=1400)
        try:
            cards   = _parse_flashcards(raw)
            display = _render_flashcards(cards, query)
            if len(cards) < 10:
                display += f"\n\n*Note: {len(cards)}/10 cards generated — try a more specific topic*"
        except Exception as err:
            display = f"## 🃏 Flashcards: *{query}*\n\n*Could not parse ({err})*\n\n{raw}"
        return display, _fmt_sources(results)
    except Exception as e:
        return f"❌ {e}", ""


def handle_exam(query, k):
    if not query.strip(): return "Please enter a topic.", ""
    try:
        db = _load_index()
        results = db.search(query, top_k=int(k))
        raw     = _generate(_exam_prompt(_fmt_context(results), query), max_new_tokens=1400)
        output  = _fix_exam_format(raw)
        return output, _fmt_sources(results)
    except Exception as e:
        return f"❌ {e}", ""


def handle_eli5(query, k):
    if not query.strip(): return "Please enter a concept.", ""
    try:
        db = _load_index()
        results = db.search(query, top_k=int(k))
        output  = _generate(_eli5_prompt(_fmt_context(results), query), max_new_tokens=700)
        return output, _fmt_sources(results)
    except Exception as e:
        return f"❌ {e}", ""


def handle_gap(question, student_ans, correct_ans, k):
    if not all([question.strip(), student_ans.strip(), correct_ans.strip()]):
        return "Please fill in all three fields.", ""
    try:
        db = _load_index()
        results = db.search(f"{question} {correct_ans}", top_k=int(k))
        prompt  = f"""A student answered an exam question incorrectly.

Question: {question}
Student answered: {student_ans}
Correct answer: {correct_ans}

Using ONLY the lecture slides below, write a clear re-explanation with these sections:

## Why the correct answer is right
Explain with a slide citation [Source N].

## Why the student answer was wrong
Be specific but constructive.

## Key concept to remember
One sentence.

## Memory aid
An analogy or tip to remember this.

Slides:
{_fmt_context(results)}

Re-explanation:"""
        output = _generate(prompt, max_new_tokens=600)
        return output, _fmt_sources(results)
    except Exception as e:
        return f"❌ {e}", ""


def handle_upload(files):
    if not files: return "No files selected."
    try:
        import pdfplumber
    except ImportError:
        return "pdfplumber not installed."
    try:
        db = _load_index()
        total, names = 0, []
        for file in files:
            path = Path(file if isinstance(file, str) else file.name)
            texts, metas = [], []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    raw = (page.extract_text() or "").strip()
                    if raw:
                        texts.append(raw)
                        metas.append({
                            "name": path.stem, "slide": i+1, "lecture_num": None,
                            "source": str(path), "filename": path.name,
                            "filetype": "pdf", "is_scanned": False,
                            "char_count": len(raw), "chunk_id": f"upload_{path.stem}_{i+1}",
                        })
            db.add_texts(texts, metas)
            total += len(texts)
            names.append(f"{path.name} ({len(texts)} slides)")
        return (
            f"✅ Added {total} slides from {len(files)} file(s):\n"
            + "\n".join(f"  • {n}" for n in names)
            + f"\n\nTotal index size: {len(db.chunks)} chunks"
        )
    except Exception as e:
        return f"❌ Upload failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.tab-nav button { font-size: 15px !important; padding: 10px 18px !important; }
.sources-box    { border-left: 4px solid #0D9488; padding: 12px 16px;
                  border-radius: 6px; font-size: 13px; margin-top: 8px; }
.output-box     { min-height: 280px; }
footer          { display: none !important; }
"""

HEADER = """
# 📚 SlideScholar
### AI Study Assistant — STATGR5293 · GenAI Using LLMs · Spring 2026

Powered by **Mistral-7B-Instruct** + **FAISS** retrieval over your actual lecture slides.
All outputs are grounded in course content — not generic AI responses.

> ⏱️ First generation may take 30–60s while the model warms up on HuggingFace servers.
"""


def build_app():
    with gr.Blocks(css=CSS, title="SlideScholar") as app:
        gr.Markdown(HEADER)

        with gr.Row():
            k_slider = gr.Slider(
                minimum=3, maximum=15, value=8, step=1,
                label="Slides to retrieve (k)",
                info="More slides = richer context but slower. Default 8 works well.",
            )

        with gr.Tab("📝 Study Guide"):
            gr.Markdown("Get structured notes with citations grounded in your lecture slides.")
            sg_query   = gr.Textbox(label="Topic or question",
                                    placeholder="e.g.  attention mechanism and transformers", lines=2)
            sg_btn     = gr.Button("Generate Study Guide", variant="primary")
            sg_output  = gr.Markdown(elem_classes=["output-box"])
            sg_sources = gr.Markdown(elem_classes=["sources-box"])
            sg_btn.click(handle_study_guide, [sg_query, k_slider], [sg_output, sg_sources])

        with gr.Tab("🃏 Flashcards"):
            gr.Markdown("Generate 10 Q&A flashcard pairs from your slides.")
            fc_query   = gr.Textbox(label="Topic",
                                    placeholder="e.g.  gradient descent and optimization", lines=2)
            fc_btn     = gr.Button("Generate Flashcards", variant="primary")
            fc_output  = gr.Markdown(elem_classes=["output-box"])
            fc_sources = gr.Markdown(elem_classes=["sources-box"])
            fc_btn.click(handle_flashcards, [fc_query, k_slider], [fc_output, fc_sources])

        with gr.Tab("📋 Practice Exam"):
            gr.Markdown("Generate a practice exam: 5 MCQs + 3 short-answer questions with answer key.")
            pe_query   = gr.Textbox(label="Topic",
                                    placeholder="e.g.  transformer architecture and self-attention", lines=2)
            pe_btn     = gr.Button("Generate Exam", variant="primary")
            pe_output  = gr.Markdown(elem_classes=["output-box"])
            pe_sources = gr.Markdown(elem_classes=["sources-box"])
            pe_btn.click(handle_exam, [pe_query, k_slider], [pe_output, pe_sources])

        with gr.Tab("💡 ELI5"):
            gr.Markdown("Explain a complex concept in simple terms using your lecture slides.")
            e5_query   = gr.Textbox(label="Concept",
                                    placeholder="e.g.  what is the attention mechanism?", lines=2)
            e5_btn     = gr.Button("Explain Simply", variant="primary")
            e5_output  = gr.Markdown(elem_classes=["output-box"])
            e5_sources = gr.Markdown(elem_classes=["sources-box"])
            e5_btn.click(handle_eli5, [e5_query, k_slider], [e5_output, e5_sources])

        with gr.Tab("🎯 Gap Analysis"):
            gr.Markdown(
                "Got a question wrong? Enter the question, your answer, and the correct answer "
                "— SlideScholar explains your mistake using your actual slides."
            )
            gap_q   = gr.Textbox(label="Exam question",
                                  placeholder="e.g.  What does softmax(QKT/sqrt(dk)) * V compute?", lines=2)
            gap_s   = gr.Textbox(label="Your answer", lines=2)
            gap_c   = gr.Textbox(label="Correct answer", lines=2)
            gap_btn = gr.Button("Explain My Mistake", variant="primary")
            gap_output  = gr.Markdown(elem_classes=["output-box"])
            gap_sources = gr.Markdown(elem_classes=["sources-box"])
            gap_btn.click(handle_gap, [gap_q, gap_s, gap_c, k_slider],
                          [gap_output, gap_sources])

        with gr.Tab("📂 Upload Slides"):
            gr.Markdown(
                "Upload additional PDF lecture slides to extend the knowledge base.\n\n"
                "> **Note:** Text is extracted directly — no vision model on Spaces. "
                "For full multimodal ingestion, run Notebook 1 in Colab and re-upload "
                "`chunks.json` and `slidescholar.faiss`."
            )
            upload_files  = gr.File(label="Upload PDF files", file_count="multiple",
                                    file_types=[".pdf"], type="filepath")
            upload_btn    = gr.Button("Add to Index", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=5)
            upload_btn.click(handle_upload, [upload_files], [upload_status])

        gr.Markdown(
            "---\n"
            "*SlideScholar · STATGR5293 · GenAI Using LLMs · Spring 2026 · Columbia University*"
        )

    return app


if __name__ == "__main__":
    print("Pre-loading index...")
    try:
        _load_index()
        print(f"Index ready — {_db.index.ntotal} vectors")
    except Exception as e:
        print(f"Warning: {e}")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)