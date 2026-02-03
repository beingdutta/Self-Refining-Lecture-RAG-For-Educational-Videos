# 🎓 Lecture-RAG for Educational Videos  
### Grounded Video Question Answering with Self-Refinement

> **TL;DR**  
> Lecture videos (slides + blackboard + face cam) break standard Video-LLMs.  
> Lecture-RAG is a grounding-aware Video-RAG framework that reduces hallucinations and supports algorithmic reasoning in educational videos.

---

## 🚨 Motivation

Most Video-Language Models are designed for:
- short clips,
- natural scenes,
- action-centric benchmarks.

**Lecture videos are fundamentally different.**

They contain:
- dense slides and equations,
- handwritten blackboard content,
- algorithm pseudocode,
- long durations with sparse visual change.

As a result, existing systems fail in two major ways:

### ❌ Hallucination
Models answer confidently from **prior knowledge**, even when:
- the relevant slide is not sampled,
- the video is blank or irrelevant.

### ❌ Over-Abstention
When strict grounding is enforced, models respond with:

> *“The answer cannot be determined from the video.”*

—even when answers are **logically derivable** from steps or equations shown in the lecture.

---

## 💡 Key Insight

> **Grounding in educational videos is not binary.**

Lecture QA requires distinguishing between:

| Grounding Type | Example | Action |
|---------------|--------|--------|
| **Explicit** | “What is written on the slide?” | Answer |
| **Derivable from steps** | “Why initialize keys to ∞ in Prim’s algorithm?” | Answer |
| **Theoretical / external** | “Why does Prim always produce an MST?” | Abstain |

Most existing approaches collapse everything into *supported vs unsupported*, which breaks algorithmic reasoning.

---

## 🧠 What is Lecture-RAG?

**Lecture-RAG** is a **Grounding-Aware Video RAG framework** tailored for educational videos.

It combines:
- OCR-based evidence extraction,
- query-aware retrieval over lecture content,
- iterative self-refinement with grounding feedback.

The goal is to:
- prevent hallucinations,
- avoid unnecessary abstention,
- support algorithmic and procedural reasoning.

---

## 🧩 Core Components

### 1️⃣ OCR-First Evidence Modeling
- OCR is treated as **primary grounding evidence**.
- The model is restricted to:
  - OCR text
  - clearly visible visual content
- External knowledge is disallowed unless **derivable from shown steps**.

---

### 2️⃣ Query-Aware OCR Retrieval
- OCR is extracted from uniformly sampled frames.
- A **hybrid retrieval module** (semantic + lexical) selects OCR segments relevant to the question.
- Removes noise from:
  - instructor bios,
  - course outlines,
  - unrelated slides.

---

### 3️⃣ Grounding-Aware Self-Refinement
Inspired by **SELF-REFINE**, adapted to multimodal grounding.

Each iteration consists of:
1. Answer generation
2. Grounding feedback classification
3. Answer refinement

Answers are classified as:
- `SUPPORTED`
- `DERIVABLE_FROM_STEPS`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`

This enables explanation-based answers without hallucination.

---

### 4️⃣ Robust Failure Handling
- On black-screen or irrelevant videos, the system **correctly abstains**.
- Prevents confident but ungrounded outputs.

---

## 🧠 Architecture Overview

```text
Video
├─ Uniform frame sampling (OCR-oriented)
├─ OCR extraction
├─ Query-aware OCR retrieval
├─ Grounded Answer Generation (Qwen2.5-VL / LLaVA / mPLUG-Owl)
├─ Grounding Feedback
└─ Iterative Self-Refinement
↓
Final Grounded Answer
```
---

## 📁 Repository Structure

```text
LectureRAG/
├── framework.py                        # Main pipeline (OCR + retrieval + refinement)
├── hybrid_search.py                    # Query-aware OCR retrieval
├── run_ocr.py                          # OCR execution script
├── nanonetOCR.py                       # OCR wrapper
├── self_refine_framework_llavaNext.py  # LLaVA-NeXT variant
├── self_refine_framework_mPlugOwl.py   # mPLUG-Owl variant
├── self_refine_framework_qwen2_5.py    # Qwen2.5-VL variant
├── frameworkocr_*.pkl                  # Cached OCR outputs
├── sampled_frames.jpeg                 # Example sampled frames
├── samples/                            # Sample lecture videos
├── README.md
```
---

## 🚀 How to Run

```bash
python self_refine_framework_qwen2_5.py
```

### Requirements

*   GPU compatible with Qwen2.5-VL / LLaVA-NeXT / mPLUG-Owl
*   Python ≥ 3.9
*   transformers, torch, decord, opencv
*   NanoNet OCR (or compatible OCR backend)

## 📚 Inspiration & Related Work

This project is inspired by:

*   **SELF-REFINE**: Iterative Refinement with Self-Feedback, NeurIPS 2023
*   **Video-RAG**: Visually-aligned Retrieval-Augmented Long Video Comprehension, NeurIPS 2025

Lecture-RAG adapts these ideas to the educational video domain, introducing grounding-aware refinement and OCR-centric retrieval.

## 🔮 Future Work

*   🔊 Automatic Speech Recognition (ASR) integration
*   🎯 Fully query-aware frame sampling
*   📊 Evaluation on educational video QA benchmarks
*   🧠 Temporal reasoning across slide transitions

## 📌 Takeaway

Lecture videos are not just another video domain.
They require OCR-aware grounding, step-based reasoning, and careful self-refinement.