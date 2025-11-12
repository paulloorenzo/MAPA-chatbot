import os
import base64
import mimetypes
import logging
import uuid
import json
import zipfile
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MAPA - Mapua AI Assistant",
    page_icon="mapua_logo.jpg",
    layout="wide"
)

# -----------------------------
# UTIL: EMBED LOGO AS BASE64 (reliable avatar)
# -----------------------------
def _logo_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        return (
            f"<img src='data:{mime};base64,{b64}' "
            f"style='width:22px;height:22px;border-radius:50%;vertical-align:middle;margin-right:8px;'>"
        )
    except Exception:
        return "🤖 "

# -----------------------------
# INITIAL CONFIG & ENV
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.ERROR)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# -----------------------------
# IMPORTING EXTERNAL EMBEDDINGS AND OTHERS (IN ZIP FILE)
# -----------------------------
def ensure_chroma_db():
    """Extract chroma_db.zip if not already extracted"""
    if not os.path.exists("chroma_db"):
        zip_path = "chroma_db.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("chroma_db")
        else:
            st.error("chroma_db.zip not found.")
            st.stop()

ensure_chroma_db()

# -----------------------------
# USER DATABASE FUNCTIONS
# -----------------------------
USERS_FILE = "users_db.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    # Default users if file doesn't exist
    return {
        "admin": "admin123",
        "user": "password123",
        "mapua": "mapua2024"
    }

def save_users(users_dict):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users_dict, f, indent=2)
        return True
    except:
        return False

# Load users at startup
USERS = load_users()

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "history" not in st.session_state:
    st.session_state.history = []  # active chat messages

# Multi-chat state
if "chats" not in st.session_state:
    st.session_state.chats = []        # [{id,title,history}]
if "active_chat_id" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.active_chat_id = cid
    st.session_state.chats.append({"id": cid, "title": "New chat", "history": []})

# Rename dialog state
if "renaming_chat_id" not in st.session_state:
    st.session_state.renaming_chat_id = None
if "rename_temp_title" not in st.session_state:
    st.session_state.rename_temp_title = ""

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False
if "context_menu_chat_id" not in st.session_state:
    st.session_state.context_menu_chat_id = None

# Sign up mode state
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# -----------------------------
# LANDING PAGE
# -----------------------------
def landing_page():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #C8102E 0%, #8B0000 100%); }
        div[data-testid="stButton"] button {
            background-color: white;
            color: #C8102E;
            border: 2px solid white;
            font-weight: bold;
            border-radius: 50px;
            padding: 12px 40px;
            font-size: 16px;
        }
        div[data-testid="stButton"] button:hover {
            background-color: rgba(255, 255, 255, 0.9);
            color: #8B0000;
            border: 2px solid white;
            transform: scale(1.05);
            transition: all 0.3s ease;
        }
        /* Remove white background from logo */
        .stImage {
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = "mapua_logo.jpg"
        if os.path.exists(logo_path):
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                st.image(logo_path, use_container_width=True)
        else:
            st.markdown(
                "<div style='text-align:center;font-size:80px;color:white;'>🤖</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: #FFB81C; font-size: 2.5em; margin-bottom: 5px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                Viva! I'm MAPA.
            </h1>
            <p style='color: #FFB81C; font-size: 2.5em; margin: 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                Your personal AI assistant
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_x, col_y, col_z = st.columns([1.5, 1, 1.5])
        with col_y:
            if st.button("Get Started ➜", use_container_width=True):
                st.session_state.page = "login"
                st.session_state.show_signup = False
                st.rerun()

        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <span style='color: rgba(255,255,255,0.7); margin: 0 15px;'>Privacy</span>
            <span style='color: rgba(255,255,255,0.5); margin: 0 5px;'>•</span>
            <span style='color: rgba(255,255,255,0.7); margin: 0 15px;'>Terms</span>
            <span style='color: rgba(255,255,255,0.5); margin: 0 5px;'>•</span>
            <span style='color: rgba(255,255,255,0.7); margin: 0 15px;'>Help</span>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# LOGIN PAGE WITH SIGN UP
# -----------------------------
def login_page():
    st.markdown("""
    <style>
        .stApp { background: white; }
        
        /* Center the tabs */
        [data-testid="stHorizontalBlock"] {
            justify-content: center !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            justify-content: center;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 30px;
            font-size: 16px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        logo_path = "mapua_logo.jpg"
        if os.path.exists(logo_path):
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                st.image(logo_path, use_container_width=True)

        # Toggle between Sign In and Sign Up
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        # SIGN IN TAB
        with tab1:
            st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #C8102E; margin-bottom: 5px;'>Sign In</h1>
                <p style='color: #666; margin-bottom: 30px;'>Access your Mapua AI Assistant</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                col_a, col_b = st.columns(2)
                with col_a:
                    submit = st.form_submit_button("Login", use_container_width=True, type="primary")
                with col_b:
                    back = st.form_submit_button("Back", use_container_width=True)
                if back:
                    st.session_state.page = "landing"
                    st.rerun()
                if submit:
                    # Reload users to get latest data
                    current_users = load_users()
                    
                    if username in current_users and current_users[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.page = "chatbot"
                        st.success(f"✅ Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")


        
        # SIGN UP TAB
        with tab2:
            st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #C8102E; margin-bottom: 5px;'>Create Account</h1>
                <p style='color: #666; margin-bottom: 30px;'>Join the Mapua AI Assistant</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("signup_form"):
                new_username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
                new_password = st.text_input("Password", type="password", placeholder="Choose a password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="signup_confirm")
                
                st.markdown("<small style='color: #666;'>Password must be at least 6 characters long</small>", unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    signup_submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                with col_b:
                    signup_back = st.form_submit_button("Back", use_container_width=True)
                
                if signup_back:
                    st.session_state.page = "landing"
                    st.rerun()
                
                if signup_submit:
                    # Reload users to get latest data
                    current_users = load_users()
                    
                    # Validation
                    if not new_username or not new_password:
                        st.error("❌ Username and password are required")
                    elif len(new_username) < 3:
                        st.error("❌ Username must be at least 3 characters long")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif new_username in current_users:
                        st.error("❌ Username already exists. Please choose another.")
                    else:
                        # Create new account
                        current_users[new_username] = new_password
                        if save_users(current_users):
                            st.success(f"✅ Account created successfully! Welcome, {new_username}!")
                            # Auto login after signup
                            st.session_state.authenticated = True
                            st.session_state.username = new_username
                            st.session_state.page = "chatbot"
                            st.rerun()
                        else:
                            st.error("❌ Error saving account. Please try again.")

# -----------------------------
# LOGOUT
# -----------------------------
def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.history = []
    st.session_state.page = "landing"
    st.rerun()

# -----------------------------
# DELETE ACCOUNT
# -----------------------------
def delete_account(username):
    """Delete user account from database"""
    current_users = load_users()
    if username in current_users:
        del current_users[username]
        if save_users(current_users):
            return True
    return False

# -----------------------------
# LOAD PDFs
# -----------------------------
def load_local_pdfs(paths):
    documents = []
    for pdf_path in paths:
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception:
                pass
    return documents

# -----------------------------
# EMBEDDINGS & VECTORSTORE
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_vectorstore(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    split_docs = splitter.split_documents(_docs)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
    return vectorstore

# -----------------------------
# ROUTING
# -----------------------------
if st.session_state.page == "landing":
    landing_page()
    st.stop()
if st.session_state.page == "login":
    login_page()
    st.stop()
if not st.session_state.authenticated:
    st.session_state.page = "landing"
    st.rerun()

# -----------------------------
# CHATBOT PAGE
# -----------------------------
st.markdown("<style>.stApp { background: white; }</style>", unsafe_allow_html=True)

# Load PDFs (only once)
if not st.session_state.pdfs_loaded:
    pdfs = ["qa_data.pdf", "llama2-deep-dataset.pdf"]
    data = load_local_pdfs(pdfs)
    if data:
        vectorstore = build_vectorstore(data)
        st.session_state.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        st.session_state.pdfs_loaded = True

# -----------------------------
# FIXED SIDEBAR STYLING
# -----------------------------
with st.sidebar:
    st.markdown("""
    <style>
      /* Sidebar container - red gradient background */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #C8102E 0%, #8B0000 100%) !important;
      }
      
      /* Remove all white backgrounds and padding */
      [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        background: transparent !important;
      }
      
      [data-testid="stSidebar"] .element-container {
        background: transparent !important;
        margin-bottom: 0 !important;
      }
      
      [data-testid="stSidebar"] [data-testid="column"] {
        background: transparent !important;
      }
      
      /* Username tile - frosted glass effect */
      .usernameTile {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      /* Section titles - white with transparency */
      .menuSectionTitle {
        font-size: 11px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 16px 8px 12px;
      }
      
      /* All buttons - semi-transparent white on red */
      [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-weight: 600;
        padding: 12px 14px;
        border-radius: 10px;
        font-size: 14px;
        transition: all 0.2s ease;
      }
      
      [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
      }
      
      /* Text input for rename - white background */
      [data-testid="stSidebar"] .stTextInput input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a1a1a !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
      }
      
      [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: white !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.3) !important;
      }
      
      [data-testid="stSidebar"] .stTextInput label {
        color: white !important;
      }
      
      /* Hide Streamlit branding */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Username tile
    st.markdown(f"<div class='usernameTile'>👤 {st.session_state.username}</div>", unsafe_allow_html=True)

    # New Chat button
    if st.button("➕  New Chat", key="btn_new_chat", use_container_width=True):
        cid = str(uuid.uuid4())
        st.session_state.chats.append({"id": cid, "title": "New chat", "history": []})
        st.session_state.active_chat_id = cid
        st.session_state.history = []
        st.rerun()

    # Section title
    st.markdown("<div class='menuSectionTitle'>Recent Chats</div>", unsafe_allow_html=True)

    # Chats list
    for idx, chat in enumerate(st.session_state.chats[::-1]):
        is_active = chat["id"] == st.session_state.active_chat_id
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            if st.button(
                f"💬 {chat.get('title') or 'New chat'}", 
                key=f"chat_{chat['id']}", 
                use_container_width=True
            ):
                st.session_state.active_chat_id = chat["id"]
                st.session_state.history = chat["history"][:]
                st.session_state.context_menu_chat_id = None
                st.rerun()
        
        with col2:
            if st.button("⋮", key=f"menu_{chat['id']}", use_container_width=True):
                if st.session_state.context_menu_chat_id == chat["id"]:
                    st.session_state.context_menu_chat_id = None
                else:
                    st.session_state.context_menu_chat_id = chat["id"]
                st.rerun()
        
        # Context menu
        if st.session_state.context_menu_chat_id == chat["id"]:
            rename_col, delete_col = st.columns(2)
            
            with rename_col:
                if st.button("✏️ Rename", key=f"ctx_rename_{chat['id']}", use_container_width=True):
                    st.session_state.renaming_chat_id = chat["id"]
                    st.session_state.rename_temp_title = chat.get("title", "")
                    st.session_state.context_menu_chat_id = None
                    st.rerun()
            
            with delete_col:
                if st.button("🗑️ Delete", key=f"ctx_delete_{chat['id']}", use_container_width=True):
                    st.session_state.chats = [c for c in st.session_state.chats if c["id"] != chat["id"]]
                    if not st.session_state.chats:
                        new_id = str(uuid.uuid4())
                        st.session_state.chats.append({"id": new_id, "title": "New chat", "history": []})
                        st.session_state.active_chat_id = new_id
                        st.session_state.history = []
                    else:
                        st.session_state.active_chat_id = st.session_state.chats[-1]["id"]
                        st.session_state.history = [*st.session_state.chats[-1]["history"]]
                    st.session_state.context_menu_chat_id = None
                    st.rerun()

        # Rename input
        if st.session_state.renaming_chat_id == chat["id"]:
            new_title = st.text_input(
                "Rename chat", 
                key=f"rename_input_{chat['id']}",
                value=st.session_state.rename_temp_title or chat.get("title",""),
                label_visibility="collapsed",
                placeholder="Enter new title..."
            )
            save_col, cancel_col = st.columns(2)
            with save_col:
                if st.button("💾 Save", key=f"save_{chat['id']}", use_container_width=True):
                    title = (new_title or "").strip()
                    if not title:
                        for m in chat["history"]:
                            if "user" in m:
                                title = m["user"].splitlines()[0][:42]
                                break
                        if not title:
                            title = "New chat"
                    chat["title"] = title if len(title) <= 48 else (title[:47] + "…")
                    st.session_state.renaming_chat_id = None
                    st.session_state.rename_temp_title = ""
                    st.rerun()
            with cancel_col:
                if st.button("✖️ Cancel", key=f"cancel_{chat['id']}", use_container_width=True):
                    st.session_state.renaming_chat_id = None
                    st.session_state.rename_temp_title = ""
                    st.rerun()

    # Logout button
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    if st.button("🚪  Logout", use_container_width=True):
        logout()
    
    # Delete Account button
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Delete Account", use_container_width=True):
        st.session_state.show_delete_confirmation = True
        st.rerun()
    
    # Delete confirmation dialog
    if st.session_state.get("show_delete_confirmation", False):
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.15); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);'>
            <p style='color: white; margin: 0; font-size: 13px; text-align: center;'>⚠️ Are you sure?</p>
        </div>
        """, unsafe_allow_html=True)
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Yes, Delete", use_container_width=True):
                if delete_account(st.session_state.username):
                    st.session_state.show_delete_confirmation = False
                    st.success("Account deleted successfully!")
                    logout()
                else:
                    st.error("Failed to delete account")
        with col_cancel:
            if st.button("✗ Cancel", use_container_width=True):
                st.session_state.show_delete_confirmation = False
                st.rerun()

# -----------------------------
# WELCOME HEADER
# -----------------------------
if len(st.session_state.history) == 0:
    st.markdown("""
    <div style="min-height:58vh; display:flex; align-items:center; justify-content:center;">
        <div style="padding:28px 22px; border-radius:18px; background:white; max-width:900px; width:100%;">
            <h1 style="margin:0 0 6px 0; font-size:40px; line-height:1.15; font-weight:800; color:#C8102E; text-align:center;">
                Hello, Mapúan!
            </h1>
            <p style="margin:0; font-size:16px; color:#666; text-align:center;">
                I am <strong>Mapa</strong>, how can I help you?
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for chat in st.session_state.history:
        if "user" in chat:
            st.markdown(f"""
            <div style='background: #f0f0f0; color: #1a1a1a; padding: 15px 20px; 
                        border-radius: 20px 20px 5px 20px; 
                        margin: 10px 0 10px auto; max-width: 80%; text-align: right;
                        border: 1px solid #d0d0d0;'>
                <strong>👤 You:</strong> {chat['user']}
            </div>
            """, unsafe_allow_html=True)
        if "assistant" in chat:
            logo_html = _logo_data_uri("mapua_logo.jpg")
            st.markdown(f"""
            <div style='background: white; color: #1a1a1a; padding: 15px 20px; 
                        border-radius: 20px 20px 20px 5px; margin: 10px auto 10px 0; 
                        max-width: 80%; border: 1px solid #e2e8f0;'>
                <strong>{logo_html}MAPA:</strong> {chat['assistant']}
            </div>
            """, unsafe_allow_html=True)


# -----------------------------
# HELPERS
# -----------------------------
def _sync_active_chat_to_store():
    for c in st.session_state.chats:
        if c["id"] == st.session_state.active_chat_id:
            c["history"] = st.session_state.history[:]
            if c["title"] == "New chat":
                for m in c["history"]:
                    if "user" in m:
                        t = m["user"].strip().splitlines()[0]
                        c["title"] = (t[:48] + "…") if len(t) > 49 else t
                        break
            break

# -----------------------------
# CHAT INPUT + RAG
# -----------------------------
query = st.chat_input("Ask MAPA")

if query and st.session_state.retriever:
    st.session_state.history.append({"user": query})
    _sync_active_chat_to_store()

    system_prompt = (
        "You are MAPA, an AI assistant for Mapua University. "
        "Use the retrieved context to answer concisely. "
        "If you don't know, say you don't know.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
    rag_chain = (
        {"context": st.session_state.retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    try:
        with st.spinner(" Thinking..."):
            response = rag_chain.invoke(query)
            st.session_state.history.append({"assistant": response})
            _sync_active_chat_to_store()
            st.rerun()
    except Exception as e:
        st.error("⚠️ Error while generating response.")
        logging.error(e)
elif query and not st.session_state.retriever:
    st.error("⚠️ Knowledge base is not ready. Please ensure PDF files are loaded correctly.")