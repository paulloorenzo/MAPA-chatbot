# MAPA Chatbot — Generative Advising in Higher Education

**MAPA Chatbot** is an AI-powered academic advising assistant designed to help students access university information efficiently through conversational interaction.  
The system uses **Retrieval-Augmented Generation (RAG)** and integrates **Gemini 2.5 Flash API** within a **Streamlit-deployed web interface** to provide grounded, context-aware responses from official Mapúa University resources.

---

## Project Overview
This repository documents the iterative development of the MAPA Chatbot system under an Agile–Scrum framework.
Each sprint focuses on progressive deliverables, improving the system’s modularity, maintainability, and performance.

---

## System Features 
- **Streamlit-based Chat UI**: Provides an interactive conversational experience through a user-friendly interface.
- **Gemini 2.5 Flash API Integration**: Powers the chatbot’s language understanding and generative responses.
- **Retrieval-Augmented Generation (RAG)**: Grounds responses in institutionally verified sources for factual accuracy.
- **Optimized Data Pipeline**: Handles document ingestion, chunking, embedding, and storage in a searchable vector database.
- **Dependency Injection Architecture**: Ensures modularity, scalability, and simplified testing by decoupling core components.
---

## System Architecture
The system follows a **two-pipeline design** that operate together to ensure accurate, data-grounded responses:
1. **Offline Knowledge Base Builder**  
   - Collects and preprocesses official university documents.
   - Generates embeddings and metadata for search optimization.
   - Stores processed data in a persistent vector database.

2. **Real-Time Query Processor**  
   - Transforms user input into vector representations.
   - Performs similarity search to retrieve relevant document chunks.
   - Constructs contextual prompts and queries the Gemini 2.5 Flash API.
   - Delivers coherent and contextually aligned responses via Streamlit.

---

## Data Model
The **Entity-Relationship Diagram (ERD)** defines four main entities that organizes user sessions, chatbot messages, and the institutional FAQ knowledge base.:

| Entity | Description |
|--------|--------------|
| **STUDENT** | Manages user authentication and demographics. |
| **CONVERSATION** | Tracks chat sessions linked to each student. |
| **MESSAGE** | Logs message pairs (user query and chatbot response) within a conversation. |
| **FAQ_DATABASE** | Contains verified academic and administrative documents used for RAG retrieval.|

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
- Includes offline ingestion (data ingestion) and online (query retrieval) pipelines.
- Embeddings stored in a searchable vector DB.
- Gemini API must process contextualized prompts.

### 2. **Functional Requirements**
- Understands English (and optionally Tagalog) queries.
- Contextually grounded answers with source citations.
- <3s average latency and multi-turn context retention.

### 3. **Usability & Design**
- UI aligns with Mapúa University branding.
- Responsive, accessible chat interface with bubbles and timestamps.
- Maintains clear distinction between user and chatbot responses.

### 4. **Performance & Testing**
- Stable under 20+ concurrent users.
- ≥80% factual accuracy (verified by academic advisors).
- Tracks MRR, nDCG, and latency metrics.

### 5. **Data Integrity & Maintenance**
- Metadata tagging for all ingested files.
- Regular re-indexing for new or updated documents.
- Secure handling of environment variables and model configuration.

### 6. **Deployment & Documentation**
- Deployable via Streamlit Cloud or localhost.
- Includes full source code, requirements.txt, and documentation.

---

## Tech Stack
| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python, Gemini 2.5 Flash API |
| **Vector Database** | Chroma |
| **Embeddings** | HuggingFace/SentenceTransformers |
| **Design** | Figma |
| **Version Control** | Git + GitHub |

---

## Repository Structure  
MAPA-chatbot/  
│  
├── app.py                     # Main Streamlit chatbot application  
├── mapa.py                    # Backup script  
├── llama2-deep-dataset.pdf    # Document data source #1  
├── qa_data.pdf                # Document data source #2  
├── mapua_logo.jpg             # Branding image for UI  
│  
├── users.json                 # User profile data (temporary storage)  
├── users_db.json              # Extended user session and history records  
│  
├── requirements.txt           # Python dependencies for Streamlit Cloud (for Streamlit)  
├── runtime.txt                # Environment/runtime version configuration   
├── README.md                  # Project overview and documentation (this file)  
└── .gitignore                 # Ignored files and folders for Git version control  
