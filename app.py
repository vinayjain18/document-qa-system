__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import shutil
import tempfile
import logging
import warnings



# Handle protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Protobuf.*')

# Set protobuf implementation if not set
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
MAX_HISTORY_LENGTH = 20
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize global variables
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name=st.secrets["OPENAI_MODEL"])

def cleanup_db():
    """Safely clean up the database directory"""
    try:
        # Wait a bit to ensure any existing connections are fully closed
        import time
        time.sleep(0.5)
        
        # Remove the directory
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            logger.info("Successfully cleaned up Chroma directory")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def process_documents(uploaded_file):
    """Process uploaded documents and update vector store"""
    try:
        documents = []
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        finally:
            os.unlink(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        # Create new DB instance and add documents
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        db.add_documents(split_docs)
        
        # Cleanup DB reference
        del db
            
        return len(split_docs)
    except Exception as e:
        logger.error(f"Error in process_documents: {e}")
        raise

def get_answer(question):
    """Get answer for a question using the RAG system"""
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise. Don't use your own knowledge and only use the above given context to answer the question.
        Answer:"""
    )

    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        results = db.similarity_search_with_score(question, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = prompt_template.format(context=context_text, question=question)
        return llm.invoke(prompt).content
    finally:
        if 'db' in locals():
            del db

def main():
    try:
        st.set_page_config(
            page_title="Document Q&A System",
            page_icon="📚",
            layout="wide"
        )

        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'docs_processed' not in st.session_state:
            st.session_state.docs_processed = False

        st.title("📚 Intelligent Document Q&A System")

        # Sidebar
        with st.sidebar:
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=["pdf"],
                help="Upload PDF files (max 10MB each)",
            )
            
            if st.button("Process Documents", type="primary"):
                try:
                    with st.spinner("Processing documents..."):
                        # Clean up existing DB before processing new documents
                        cleanup_db()
                        num_docs = process_documents(uploaded_files)
                        if num_docs > 0:
                            st.success(f"✅ Processed {num_docs} document chunks")
                            st.session_state.docs_processed = True
                        else:
                            st.error("No valid documents were processed")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    logger.error(f"Document processing error: {e}")

            # Chat history cleanup
            if st.session_state.docs_processed:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

        # Q&A section
        if st.session_state.docs_processed:
            question = st.text_input(
                "Ask your question:",
                placeholder="Enter your question here...",
                key="question_input"
            )

            if question:
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        answer = get_answer(question)
                        
                        # Update chat history
                        st.session_state.chat_history.append((question, answer))
                        if len(st.session_state.chat_history) > MAX_HISTORY_LENGTH:
                            st.session_state.chat_history.pop(0)

                    except Exception as e:
                        logger.error(f"Error generating answer: {e}")
                        st.error("An error occurred while generating the answer. Please try again.")
                    
                    # Display chat history
                    if st.session_state.chat_history:
                        st.subheader("💬 Conversation History")
                        for q, a in reversed(st.session_state.chat_history):
                            with st.container():
                                st.info(f"Question: {q}")
                                st.success(f"Answer: {a}")
                                st.divider()
        else:
            st.info("👆 Please upload and process documents to start asking questions.")

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()