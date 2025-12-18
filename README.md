# SIPY - AI-Driven Investment Planning & Optimization Engine

SIPY is a hybrid AI system designed to convert unstructured user inputs into structured, personalized investment recommendations with clear, layman-friendly explanations.

The system combines deterministic logic with AI-based reasoning to support decision-making under uncertainty, while prioritizing explainability and consistency over black-box predictions.

This repository contains the backend API and core decision logic powering SIPY.

---

## Problem Statement

Most retail investors describe their goals, constraints, and concerns in natural language—often vague, incomplete, or emotionally driven.  
However, investment decisions require structured inputs, explicit constraints, and trade-offs.

SIPY bridges this gap by:
- Interpreting unstructured text or voice inputs  
- Translating them into structured decision parameters  
- Producing explainable, personalized investment recommendations  

---

## What This System Does

At a high level, SIPY:

1. Ingests unstructured user inputs (text or voice)  
2. Translates them into structured, system-readable parameters  
3. Models personalization and constraints  
4. Designs a hyper-personalized glide path and investment strategy  
5. Constructs a portfolio with asset allocation and rationale  
6. Explains recommendations in simple, non-technical language  

The system is designed for **decision support**, not prediction alone.

---

## System Architecture (High-Level)

User Input (Text / Voice)
↓
Unstructured → Structured Translation
↓
Personalization & Constraint Modeling
↓
Glide Path & Strategy Engine
↓
Portfolio Construction
↓
Explanation Layer (Layman-Friendly)


## Unstructured → Structured Translation

Users interact with SIPY through text or voice, describing goals, timelines, and preferences in natural language.

These inputs are:
- Parsed using AI-assisted interpretation  
- Translated into structured decision parameters  
- Persisted using automated workflows (e.g., Airtable-based schemas)  

This layer acts as the bridge between human language and system logic.

---

## Personalization & Constraint Modeling

Structured inputs are used to model the user’s financial context, including:
- Time to goal  
- Risk profile  
- **Funding gap** (core personalization driver)  

The funding gap determines how aggressive or conservative the system’s recommendations can be and acts as a key constraint throughout the decision process.

---

## Glide Path & Strategy Engine

Based on personalization and constraints, SIPY:
- Designs a hyper-personalized glide path  
- Selects an appropriate investment strategy  
- Applies deterministic rules where constraints are clear  
- Uses AI-based reasoning to handle ambiguity and trade-offs  

This hybrid approach ensures control, consistency, and adaptability.

---

## Portfolio Construction

The selected strategy is translated into:
- A concrete portfolio  
- Asset allocation and weights  
- Clear rationale for each decision  

The system emphasizes explainability and coherence over opaque optimization.

---

## Explanation Layer

All outputs pass through an explanation layer designed explicitly for non-expert users.

This layer:
- Converts system decisions into simple, human-readable explanations  
- Clearly communicates *why* a recommendation was made  
- Avoids financial and technical jargon wherever possible  

Explainability is treated as a first-class product feature.

---

## Deterministic vs AI Components

**Deterministic Logic**
- Constraint validation  
- Rule-based filtering  
- Boundary conditions (risk limits, timelines, eligibility)  

**AI-Based Reasoning**
- Interpreting ambiguous user intent  
- Handling trade-offs across conflicting constraints  
- Generating natural-language explanations  

AI is used selectively, not as a black box.

---

## Why This Architecture Is Different

- **Hybrid by design** — combines rules and AI instead of relying on opaque end-to-end models  
- **Constraint-first personalization** — funding gap and time-to-goal drive decisions, not generic risk buckets  
- **Explainability-led** — recommendations are designed to be understood, not just generated  
- **Decision-focused** — optimized for real-world decision support under uncertainty  

---

## Tech Stack (Indicative)

- Python  
- FastAPI  
- AI-assisted reasoning (LLM APIs)  
- Modular rule engine  
- Structured decision schemas  
- Automation-driven data persistence  

---

## Status

This repository represents an evolving system under active iteration.  
It is intended to demonstrate **product thinking, system design, and applied AI workflows**, rather than production hardening.

