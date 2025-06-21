import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 Smart PDF Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid transparent;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .upload-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {'pages': 0, 'chunks': 0, 'files': 0}

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Smart PDF Chat Assistant</h1>
    <p>Upload your documents and have intelligent conversations with AI</p>
    <p><em>Created by Ansh Jha</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>⚙️ Configuration</h3>
        <p>Set up your AI assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input
    api_key = st.text_input(
        "🔑 Enter your Groq API key:",
        type="password",
        help="Enter your Groq API key to enable AI functionality"
    )
    
    # Session management
    session_id = st.text_input(
        "💬 Session ID",
        value="default_session",
        help="Unique identifier for your chat session"
    )
    
    # Model selection
    model_options = {
        "Llama 3.1 8B (Recommended)": "llama-3.1-8b-instant",
        "Llama 3.1 70B (Advanced)": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768"
    }
    
    selected_model = st.selectbox(
        "🧠 Choose AI Model",
        options=list(model_options.keys()),
        help="Select the AI model for processing"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 1000, 8000, 5000)
        chunk_overlap = st.slider("Chunk Overlap", 100, 1000, 500)
        max_sentences = st.slider("Max Response Sentences", 1, 10, 3)
    
    # Statistics
    if st.session_state.document_stats['files'] > 0:
        st.markdown("### 📊 Document Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Files", st.session_state.document_stats['files'])
        with col2:
            st.metric("📑 Pages", st.session_state.document_stats['pages'])
        with col3:
            st.metric("🧩 Chunks", st.session_state.document_stats['chunks'])

# Main content area
if not api_key:
    st.markdown("""
    <div class="upload-section">
        <h3>🔑 API Key Required</h3>
        <p>Please enter your Groq API key in the sidebar to get started!</p>
        <p>You can obtain your API key from the Groq platform</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Initialize LLM
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_options[selected_model],
            temperature=0.1
        )
        
        # Initialize embeddings with proper device handling
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    except Exception as e:
        st.error(f"❌ Error initializing AI model: {str(e)}")
        st.stop()

    # Document upload section
    st.markdown("### 📁 Upload Your Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with their content"
    )
    
    # Process uploaded files
    if uploaded_files and not st.session_state.processing_complete:
        with st.spinner('🔄 Processing your documents...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            documents = []
            total_pages = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f'Processing: {uploaded_file.name}')
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Save temporary file
                temp_pdf = f"./temp_{i}.pdf"
                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                # Load and process PDF
                try:
                    loader = PyPDFLoader(temp_pdf)
                    docs = loader.load()
                    documents.extend(docs)
                    total_pages += len(docs)
                    
                    # Clean up temp file
                    os.remove(temp_pdf)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            if documents:
                # Split documents
                status_text.text('🔪 Splitting documents into chunks...')
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                status_text.text('🧠 Creating knowledge base...')
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings
                )
                retriever = vectorstore.as_retriever()
                
                # Store in session state
                st.session_state.retriever = retriever
                st.session_state.processing_complete = True
                st.session_state.document_stats = {
                    'files': len(uploaded_files),
                    'pages': total_pages,
                    'chunks': len(splits)
                }
                
                progress_bar.progress(1.0)
                status_text.text('✅ Processing complete!')
                
                st.success(f"🎉 Successfully processed {len(uploaded_files)} files with {total_pages} pages!")
                time.sleep(1)
                st.rerun()
    
    # Chat interface
    if hasattr(st.session_state, 'retriever') and st.session_state.processing_complete:
        st.markdown("### 💬 Chat with Your Documents")
        
        # Setup RAG chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, contextualize_q_prompt
        )
        
        system_prompt = (
            f"You are an intelligent assistant for question-answering tasks. "
            f"Use the following pieces of retrieved context to answer "
            f"the question accurately and comprehensively. If you don't know the answer, "
            f"say that you don't know. Use maximum {max_sentences} sentences and keep the "
            f"answer informative yet concise. Always provide specific details when available."
            f"\n\n"
            f"{{context}}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>🙋‍♂️ You:</strong><br>
                        {message['content']}
                        <br><small>{message['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message['content']}
                        <br><small>{message['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything about your documents:",
                key="user_input",
                placeholder="e.g., What are the main topics discussed?",
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.button("🚀 Ask", use_container_width=True)
        
        # Sample questions
        st.markdown("**💡 Try these sample questions:**")
        sample_questions = [
            "What are the main topics covered?",
            "Can you summarize the key points?",
            "What are the conclusions?",
            "Are there any recommendations?"
        ]
        
        cols = st.columns(4)
        for i, question in enumerate(sample_questions):
            with cols[i]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    user_input = question
                    ask_button = True
        
        # Process user input
        if (user_input and ask_button) or (user_input and st.session_state.get('user_input')):
            with st.spinner('🤔 Thinking...'):
                try:
                    # Get response
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    # Add to chat history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input,
                        'timestamp': timestamp
                    })
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'timestamp': timestamp
                    })
                    
                    # Clear input and rerun
                    if 'user_input' in st.session_state:
                        del st.session_state['user_input']
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
        
        # Chat management buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if session_id in st.session_state.store:
                    del st.session_state.store[session_id]
                st.rerun()
        
        with col2:
            if st.button("📥 Export Chat", use_container_width=True):
                chat_data = {
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'chat_history': st.session_state.chat_history
                }
                st.download_button(
                    "💾 Download Chat History",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_history_{session_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🔄 Reset Documents", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.document_stats = {'pages': 0, 'chunks': 0, 'files': 0}
                if 'retriever' in st.session_state:
                    del st.session_state.retriever
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚀 Built with Streamlit, LangChain, and Groq AI</p>
    <p>Made with ❤️ for intelligent document processing</p>
</div>
""", unsafe_allow_html=True)