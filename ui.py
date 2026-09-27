#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import json
import csv
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from rag_core import RAGSystem
## Removed unused import: db_manager
import bcrypt
import streamlit as st
import time
import base64
from database import DatabaseManager
from user_activity_logger import UserActivityLogger


if 'db_manager' not in st.session_state or st.session_state['db_manager'] is None:
    st.session_state['db_manager'] = DatabaseManager("sqlite:///rag_system.db")
db_manager = st.session_state['db_manager']

# Page configuration
st.set_page_config(
    page_title="RAG for Public Service Documents",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and clean styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    
    .stButton > button {
        background-color: #00d4aa;
        color: #0e1117;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #00b894;
    }
    
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    
    .stFileUploader > div > div > div {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 8ss_querypx;
    }
    
    .stProgress > div > div > div > div {
        background-color: #00d4aa;
    }
    
    .main {
        padding-top: 1rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        background-color: #1e1e1e;
        color: #fafafa !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .panel {
        background: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .panel-title {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        color: #00d4aa;
        border-bottom: 2px solid #333333;
        padding-bottom: 0.5rem;
    }
    
    .confidence-meter {
        background: #262730;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .confidence-good {color: #00d4aa;}
    .confidence-warn {color: #ffa726;}
    .confidence-bad {color: #ef5350;}
    
    .warning-box {
        background-color: #2d1b1b;
        border: 1px solid #ef5350;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #1b2d1b;
        border: 1px solid #00d4aa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .action-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .action-button {
        background: #00d4aa;
        color: #0e1117;
        border: none;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.9rem;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }
    
    .action-button:hover {
        background: #00b894;
        color: #0e1117;
        text-decoration: none;
    }
    
    .secondary-button {
        background: #333333;
        color: #fafafa;
        border: 1px solid #4a4a4a;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.9rem;
        cursor: pointer;
    }
    
    .secondary-button:hover {
        background: #4a4a4a;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        # Automatically initialize the system
        with st.spinner("Initializing system..."):
            success = st.session_state.rag_system.initialize_system()
            if not success:
                st.error("❌ Failed to initialize system")
    if "user" not in st.session_state:
        st.session_state["user"] = None
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "selected_source_idx" not in st.session_state:
        st.session_state.selected_source_idx = 0

# Log feedback to CSV
def log_feedback(query, response, feedback, rating=None):
    logger = UserActivityLogger()
    user = st.session_state.get("user", "anonymous")
    logger.log_event(user=user, query=query, response=response, feedback=feedback)

# Check if system is ready for queries
def is_system_ready():
    """Check if the system has documents for querying"""
    return len(st.session_state.rag_system.documents) > 0

def get_sources_summary(documents):
    by_file = {}
    for doc in documents:
        name = doc.metadata.get('filename', 'Unknown')
        page = doc.metadata.get('page_number', 1)
        if name not in by_file:
            by_file[name] = {"pages": set(), "count": 0}
        by_file[name]["pages"].add(page)
        by_file[name]["count"] += 1
    # convert pages set to count
    return {name: {"pages": len(data["pages"]), "count": data["count"]} for name, data in by_file.items()}

def build_note_markdown(entry):
    title = f"Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    query = entry['query']
    answer = entry['response']
    is_h = entry.get('is_hallucination', False)
    conf = float(entry.get('hallucination_confidence', 0.0))
    grounding = max(0.0, min(1.0, 1.0 - conf))
    docs = entry.get('source_docs', []) or []
    citations = "\n".join([
        f"- {doc.metadata.get('filename','Unknown')} (page {doc.metadata.get('page_number','N/A')})"
        for doc in docs[:10]
    ])
    md = f"""# {title}

## Question
{query}

## Answer
{answer}

## Grounding
- Hallucination detected: {'Yes' if is_h else 'No'}
- Grounding confidence: {grounding:.2f}

## Citations
{citations if citations else '- No source documents returned'}
"""
    return md

def generate_shareable_link(query, response, confidence, is_hallucination):
    """Generate a shareable link with response data"""
    data = {
        "query": query,
        "response": response,
        "confidence": confidence,
        "is_hallucination": is_hallucination,
        "timestamp": datetime.now().isoformat()
    }
    
    # Encode data as base64
    json_str = json.dumps(data)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    
    # Create shareable URL (you can replace with your actual domain)
    base_url = "https://your-domain.com/share"
    return f"{base_url}?data={encoded}"

def render_document_upload_panel():
    logger = UserActivityLogger()
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📁 Document Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files (max 5 at a time)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload public service documents to analyze"
    )
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("You can upload a maximum of 5 files at a time.")
            return
        use_parallel = st.checkbox("🚀 Use parallel processing", value=True)
        if st.button("📤 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                stats = st.session_state.rag_system.add_documents(uploaded_files, use_parallel=use_parallel)
                user = st.session_state.get("user", "anonymous")
                for f in uploaded_files:
                    logger.log_event(user=user, document=f.name)
                if stats.get("errors"):
                    for error in stats["errors"]:
                        st.error(f"⚠️ {error}")
                if stats["new_documents"] > 0:
                    st.success(f"✅ Added {stats['new_documents']} document segments")
                    st.info(f"📚 Total documents: {stats['total_documents']}")
                    st.rerun()
                else:
                    st.info("No new documents were added")
    # Display current documents
    if st.session_state.rag_system.documents:
        st.markdown("### 📚 Current Documents")
        summary = get_sources_summary(st.session_state.rag_system.documents)
        for name, data in summary.items():
            st.markdown(f"• **{name}** — {data['pages']} pages, {data['count']} chunks")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_query_panel():
    """Render the query input panel"""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💬 Ask Questions</div>', unsafe_allow_html=True)
    
    if not is_system_ready():
        st.info("📚 Upload documents to start asking questions")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Query input
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_area(
            "Enter your question about the public service documents:",
            placeholder="e.g., What are the eligibility criteria for social benefits?",
            height=100
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("🔍 Ask Question", type="primary")
        with col2:
            if st.form_submit_button("🗑️ Clear chat"):
                st.session_state.query_history = []
                st.rerun()
    
    if submitted and user_query:
        process_query(user_query)
    
    # Display recent responses
    if st.session_state.query_history:
        st.markdown("### 📝 Recent Responses")
        for i, entry in enumerate(st.session_state.query_history[:3]):
            with st.expander(f"Q: {entry['query'][:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:**")
                st.markdown(f'<div class="chat-message">{entry["response"]}</div>', unsafe_allow_html=True)
                # Feedback buttons
                col_feedback1, col_feedback2 = st.columns([1,1])
                with col_feedback1:
                    if st.button("👍 Thumbs Up", key=f"thumbs_up_{i}"):
                        log_feedback(entry['query'], entry['response'], 'thumbs_up')
                        st.success("Thank you for your feedback!")
                with col_feedback2:
                    if st.button("👎 Thumbs Down", key=f"thumbs_down_{i}"):
                        log_feedback(entry['query'], entry['response'], 'thumbs_down')
                        st.info("Feedback noted.")
                # Action buttons for each response
                col1, col2 = st.columns(2)
                with col1:
                    note_md = build_note_markdown(entry)
                    st.download_button(
                        "📥 Download",
                        note_md,
                        file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        key=f"download_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        mime="text/markdown"
                    )
                with col2:
                    if entry.get('source_docs'):
                        st.markdown(f"📚 {len(entry['source_docs'])} sources")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_evaluation_panel():
    """Render the evaluation and confidence panel"""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 Response Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.query_history:
        st.info("Ask a question to see response analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Dropdown to select which query to analyze
    query_options = [f"{i+1}: {entry['query'][:40]}..." for i, entry in enumerate(st.session_state.query_history)]
    selected_idx = st.selectbox("Select a query to analyze:", options=list(range(len(query_options))), format_func=lambda i: query_options[i], index=0)
    entry = st.session_state.query_history[selected_idx]
    is_hallucination = entry.get('is_hallucination', False)
    confidence = float(entry.get('hallucination_confidence', 0))
    grounding = max(0.0, min(1.0, 1.0 - confidence))
    
    # Confidence meter
    st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
    st.markdown("### 🎯 Grounding Confidence")
    
    if grounding >= 0.75:
        label = "High"
        color_class = "confidence-good"
    elif grounding >= 0.45:
        label = "Medium"
        color_class = "confidence-warn"
    else:
        label = "Low"
        color_class = "confidence-bad"
    
    st.markdown(f"**{label}** <span class='{color_class}'>({grounding:.2f})</span>", unsafe_allow_html=True)
    st.progress(grounding)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hallucination warning
    if is_hallucination:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **Potential Hallucination Detected**")
        st.markdown(f"Confidence: {confidence:.2f}")
        st.markdown("Please verify this response against the source documents.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Source documents
    docs = entry.get('source_docs', []) or []
    import logging
    logging.info(f"[DEBUG] UI displaying {len(docs[:5])} source documents for this query.")
    if docs:
        st.markdown("### 📚 Source Documents")
        for i, doc in enumerate(docs[:5]):
            with st.expander(f"Source {i+1}: {doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page_number', 'N/A')})"):
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
    
    # Processing time
    processing_time = entry.get('processing_time', 0)
    st.markdown(f"**⏱️ Response Time:** {processing_time:.2f}s")
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_query(user_query):
    # Query preprocessing removed - using raw user query
    if not is_system_ready():
        st.error("⚠️ No documents available. Please upload documents first.")
        return
    logger = UserActivityLogger()
    user = st.session_state.get("user", "anonymous")
    with st.spinner("🔍 Searching documents..."):
        start_time = time.time()
        response_data = st.session_state.rag_system.answer_query(user_query)
        
        response = response_data["answer"]
        source_docs = response_data["source_documents"]
        hallucination_info = response_data.get("hallucination_info", {})
        processing_time = response_data.get("processing_time", 0)
        
        # Add to query history
        query_entry = {
            "query": user_query,
            "preprocessed_query": user_query,
            "response": response,
            "time": datetime.now().strftime("%H:%M:%S"),
            "is_hallucination": hallucination_info.get("is_hallucination", False),
            "hallucination_confidence": hallucination_info.get("confidence", 0.0),
            "processing_time": processing_time,
            "source_docs": source_docs
        }
        
        st.session_state.query_history.insert(0, query_entry)
        logger.log_event(user=user, query=user_query, response=response)

def render_main_layout():
    st.title("🏛️ RAG for Public Service Documents")
    st.markdown("---")
    
    # Three column layout
    col1, col2, col3 = st.columns([1, 1.5, 1], gap="large")
    
    with col1:
        render_document_upload_panel()
    
    with col2:
        render_query_panel()
    
    with col3:
        render_evaluation_panel()

# --- MFA Utilities ---
import secrets
import bcrypt
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from database import User

OTP_EXPIRY_SECONDS = 300  # 5 minutes
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'ragforpublicservices@gmail.com'
SMTP_PASSWORD = 'jbwh dqxj yapw piwd'

def generate_and_send_otp(user, db_manager):
    otp = str(secrets.randbelow(1000000)).zfill(6)
    otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
    expiry = datetime.now() + timedelta(seconds=OTP_EXPIRY_SECONDS)
    session = db_manager.get_session()
    db_user = session.query(User).filter(User.id == user.id).first()
    db_user.otp_hash = otp_hash
    db_user.otp_expiry = expiry
    session.commit()
    session.close()
    send_otp_email(user.email, otp)
    return True

def send_otp_email(email, otp):
    msg = MIMEText(f"Your OTP is: {otp}")
    msg['Subject'] = "Your Login OTP"
    msg['From'] = SMTP_USER
    msg['To'] = email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(msg['From'], [msg['To']], msg.as_string())

def verify_otp(user, otp, db_manager):
    session = db_manager.get_session()
    db_user = session.query(User).filter(User.id == user.id).first()
    valid = False
    if db_user.otp_expiry and datetime.now() <= db_user.otp_expiry:
        valid = bcrypt.checkpw(otp.encode(), db_user.otp_hash.encode())
    session.close()
    return valid

# --- MFA UI Panels ---
def show_login_panel():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        user = db_manager.verify_user_password(username, password)
        if user:
            st.session_state['pending_user'] = user
            st.session_state['otp_required'] = True
            generate_and_send_otp(user, db_manager)
            st.info("Credentials valid, OTP required. Check your email.")
            st.rerun()
        else:
            st.error("Invalid credentials")

def show_otp_panel():
    st.subheader("Enter OTP")
    with st.form("otp_form", clear_on_submit=True):
        otp = st.text_input("Enter the OTP sent to your email", key="otp_input")
        submitted = st.form_submit_button("Verify OTP", type="primary")
        if submitted:
            user = st.session_state.get('pending_user')
            if user and verify_otp(user, otp, db_manager):
                st.session_state['user'] = user.username
                st.session_state['otp_required'] = False
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid or expired OTP")

def main():
    apply_custom_css()
    initialize_session_state()
    if st.session_state.get("otp_required", False):
        show_otp_panel()
    elif st.session_state["user"] is None:
        show_login_panel()
        show_registration_panel()
    else:
        render_main_layout()
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 1rem;'>"
            "RAG for Public Service Documents | Built with Streamlit & LangChain"
            "</div>",
            unsafe_allow_html=True
        )

def show_registration_panel():
    st.subheader("Register")
    username = st.text_input("New Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("New Password", type="password", key="register_password")
    if st.button("Register", key="register_button"):
        import bcrypt
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user_data = {
            "username": username,
            "email": email,
            "password": hashed_pw.decode(),
            "role": "user"
        }
        user = db_manager.create_user(user_data)
        if user:
            st.success("Registration successful! Please login.")
        else:
            st.error("Registration failed. Username or email may already exist.")

if __name__ == "__main__":
    main()
