"""
Streamlit Chat Interface for RAG System
Usage: streamlit run src/apps/chat_app.py
"""

import streamlit as st
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add src to path for RAG pipeline import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    logging.error(f"RAG import error: {e}")
    RAG_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatSession:
    """Manages chat session state."""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None) -> None:
        self.messages.append({
            'role': role, 'content': content, 'timestamp': datetime.now().isoformat(),
            'sources': sources or []
        })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages.copy()
    
    def clear_messages(self) -> None:
        self.messages.clear()


class RAGChatInterface:
    """Main chat interface integrating with RAG pipeline."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.session = ChatSession()
        self.error_count = 0
        self.max_errors = 5
    
    def initialize_rag_pipeline(self) -> bool:
        try:
            if not RAG_AVAILABLE:
                return False
            if self.rag_pipeline is None:
                self.rag_pipeline = RAGPipeline()
            return True
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            return False
    
    def process_question(self, question: str) -> Tuple[str, List[Dict], bool]:
        if not question or not question.strip():
            return "Please provide a valid question.", [], False
        
        try:
            if not self.initialize_rag_pipeline():
                return "RAG pipeline not available.", [], False
            
            result = self.rag_pipeline.answer_question(question)
            answer = result.get('answer', 'No answer generated.')
            sources = result.get('retrieved_sources', [])
            self.error_count = 0
            return answer, sources, True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Question processing error: {e}")
            if self.error_count >= self.max_errors:
                return "Too many errors. Please refresh.", [], False
            return f"Error: {str(e)}", [], False
    
    def format_sources(self, sources: List[Dict]) -> str:
        if not sources:
            return "No sources available."
        
        formatted = []
        for i, source in enumerate(sources, 1):
            text = source.get('text', 'No text available')
            if len(text) > 300:
                text = text[:300] + "..."
            formatted.append(f"**Source {i}:**\n{text}")
        
        return "\n\n".join(formatted)
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        if not user_input or not user_input.strip():
            return False, "Please enter a question."
        if len(user_input.strip()) < 3:
            return False, "Question must be at least 3 characters."
        if len(user_input) > 1000:
            return False, "Question too long. Keep under 1000 characters."
        return True, ""


class StreamlitUI:
    """Streamlit UI components."""
    
    @staticmethod
    def setup_page():
        st.set_page_config(page_title="Financial Complaint Analysis Chat", page_icon="💬", layout="wide")
    
    @staticmethod
    def render_header():
        st.title("🤖 Financial Complaint Analysis Chat")
        st.markdown("Ask questions about financial complaints and get AI-powered insights with source verification.")
    
    @staticmethod
    def render_sidebar(chat_interface: RAGChatInterface):
        with st.sidebar:
            st.header("📊 Session Info")
            st.metric("Messages", len(chat_interface.session.get_messages()))
            st.metric("Errors", chat_interface.error_count)
            
            st.divider()
            st.header("🎛️ Controls")
            
            if st.button("🗑️ Clear Chat", type="secondary"):
                chat_interface.session.clear_messages()
                st.rerun()
            
            if st.button("🔄 Refresh System", type="secondary"):
                chat_interface.initialize_rag_pipeline()
                st.success("System refreshed!")
            
            st.divider()
            st.header("❓ Help")
            st.markdown("""
            **How to use:**
            1. Type your question
            2. Click 'Ask' or press Enter
            3. View response with sources
            4. Use 'Clear Chat' to restart
            
            **Examples:**
            - "What are common credit card complaints?"
            - "How do customers feel about loan services?"
            """)
    
    @staticmethod
    def render_chat_messages(chat_interface: RAGChatInterface):
        messages = chat_interface.session.get_messages()
        
        for message in messages:
            with st.chat_message(message['role']):
                st.write(message['content'])
                
                if message['role'] == 'assistant' and message.get('sources'):
                    with st.expander("📚 View Sources", expanded=False):
                        st.markdown(chat_interface.format_sources(message['sources']))
    
    @staticmethod
    def render_input_section(chat_interface: RAGChatInterface):
        user_question = st.chat_input("Ask a question about financial complaints...")
        
        if user_question:
            is_valid, error_msg = chat_interface.validate_input(user_question)
            if not is_valid:
                st.error(error_msg)
                return
            
            chat_interface.session.add_message('user', user_question)
            
            with st.chat_message('user'):
                st.write(user_question)
            
            with st.chat_message('assistant'):
                with st.spinner('🤔 Thinking...'):
                    answer, sources, success = chat_interface.process_question(user_question)
                    
                    if success:
                        st.write(answer)
                        if sources:
                            with st.expander("📚 View Sources", expanded=False):
                                st.markdown(chat_interface.format_sources(sources))
                    else:
                        st.error(answer)
                
                chat_interface.session.add_message('assistant', answer, sources)
    
    @staticmethod
    def render_footer():
        st.divider()
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
            "Built with ❤️ using Streamlit and RAG technology<br>"
            "Intelligent Complaint Analysis for Financial Services"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Main application function."""
    try:
        StreamlitUI.setup_page()
        
        if 'chat_interface' not in st.session_state:
            st.session_state.chat_interface = RAGChatInterface()
        
        chat_interface = st.session_state.chat_interface
        
        StreamlitUI.render_header()
        StreamlitUI.render_sidebar(chat_interface)
        StreamlitUI.render_chat_messages(chat_interface)
        StreamlitUI.render_input_section(chat_interface)
        StreamlitUI.render_footer()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or contact support.")


if __name__ == "__main__":
    main() 