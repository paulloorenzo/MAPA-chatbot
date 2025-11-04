# MAPA Chatbot — Generative Advising in Higher Education

**MAPA Chatbot** is an AI-powered academic advising assistant designed to help students access university information efficiently through conversational interaction.  
The system uses **Retrieval-Augmented Generation (RAG)** and integrates **Gemini 2.5 Flash API** within a **Streamlit-deployed web interface** to provide grounded, context-aware responses from official Mapúa University resources.

---

## Project Overview
This repository contains the development artifacts for **Sprint 1**, which focuses on:
- Establishing the system foundation and architecture.
- Designing the user interface (UI).
- Modeling the data structure through an Entity-Relationship Diagram (ERD).
- Defining functional and non-functional acceptance criteria.

---

## System Features (Sprint 1 Goals)
- **Streamlit-based Chat UI**: Interactive chatbot interface for real-time user interaction.
- **Gemini 2.5 Flash API Integration**: Handles natural language understanding and response generation.
- **Retrieval-Augmented Generation (RAG)**: Ensures responses are grounded in verified institutional data sources.
- **Data Pipeline**: Supports document ingestion, preprocessing, metadata tagging, and vector embedding.
- **Scalable Database Schema**: Tracks users, sessions, messages, and reference sources.

---

## System Architecture
The system follows a **two-pipeline design**:
1. **Offline Knowledge Base Builder**  
   - Document ingestion and chunking  
   - Metadata tagging  
   - Vector embedding generation  
   - Stored in a searchable vector database  

2. **Real-Time Query Processor**  
   - Converts user queries into embeddings  
   - Performs similarity search  
   - Retrieves top-N context chunks  
   - Sends the contextualized prompt to Gemini API for response generation  

---

## Data Model
The **Entity-Relationship Diagram (ERD)** defines four main entities:

| Entity | Description |
|--------|--------------|
| **STUDENT** | Manages user authentication and demographics. |
| **CONVERSATION** | Tracks chat sessions linked to each student. |
| **MESSAGE** | Logs individual dialogue turns between user and bot. |
| **FAQ_DATABASE** | Stores RAG knowledge base content and metadata. |

**Key Relationships:**
- `STUDENT` → `CONVERSATION` (1:M)  
- `CONVERSATION` → `MESSAGE` (1:M)  
- `FAQ_DATABASE` → `MESSAGE` (1:0..M, optional link for response traceability)

**ERD Reference:**  
[📘 Figma Design & ERD](https://www.figma.com/design/GjgwHn1LdInC1PegmVDFDl/SofttDev?node-id=0-1&t=sfW8zK9TY7Tviivk-1)

---

## Acceptance Criteria Summary
The system must satisfy the following high-level criteria:

### 1. **System Design & Architecture**
- Includes offline ingestion and online retrieval pipelines.
- Embeddings stored in a searchable vector DB.
- Gemini API must process contextualized prompts.

### 2. **Functional Requirements**
- Understands English (and optionally Tagalog) queries.
- Contextually grounded answers with source citations.
- <3s average latency and multi-turn context retention.

### 3. **Usability & Design**
- UI aligns with Mapúa University branding.
- Responsive, accessible chat interface with bubbles and timestamps.

### 4. **Performance & Testing**
- Stable under 20+ concurrent users.
- ≥80% factual accuracy (verified by academic advisors).
- Tracks MRR, nDCG, and latency metrics.

### 5. **Data Integrity & Maintenance**
- Metadata tagging for all ingested files.
- Periodic refresh and re-indexing of updated data.
- Guardrails for private or sensitive queries.

### 6. **Deployment & Documentation**
- Deployable via Streamlit Cloud or localhost.
- Includes full source code, requirements.txt, and documentation.

---

## Tech Stack
| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python, Gemini 2.5 Flash API |
| **Database** | Vector DB (for embeddings), relational DB (for user data) |
| **Design** | Figma |
| **Version Control** | Git + GitHub |

---

## Repository Structure  
MAPA-chatbot/  
│  
├── app/ # Streamlit application  
├── data/ # Knowledge base and preprocessed data  
├── models/ # RAG + Gemini integration modules  
├── docs/ # Documentation and design files  
├── requirements.txt # Python dependencies  
├── README.md # Project overview (this file)  
└── .gitignore  
