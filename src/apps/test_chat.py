"""
Test script for the chat interface components.
Run: python src/apps/test_chat.py
"""

import sys
import os
from unittest.mock import Mock, patch

# Add parent directories to path for RAG pipeline import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from chat_app import ChatSession, RAGChatInterface, StreamlitUI


def test_chat_session():
    """Test ChatSession functionality."""
    print("🧪 Testing ChatSession...")
    
    session = ChatSession()
    session.add_message('user', 'Hello')
    session.add_message('assistant', 'Hi there!', [{'text': 'source1'}])
    
    messages = session.get_messages()
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[1]['sources'] == [{'text': 'source1'}]
    
    session.clear_messages()
    assert len(session.get_messages()) == 0
    
    print("✅ ChatSession tests passed!")


def test_rag_chat_interface():
    """Test RAGChatInterface functionality."""
    print("🧪 Testing RAGChatInterface...")
    
    interface = RAGChatInterface()
    
    # Test input validation
    is_valid, error = interface.validate_input("")
    assert not is_valid
    assert "Please enter" in error
    
    is_valid, error = interface.validate_input("Hi")
    assert not is_valid
    assert "at least 3 characters" in error
    
    is_valid, error = interface.validate_input("What are common complaints?")
    assert is_valid
    assert error == ""
    
    # Test source formatting
    sources = [{'text': 'This is a test source document.'}]
    formatted = interface.format_sources(sources)
    assert "Source 1:" in formatted
    assert "test source document" in formatted
    
    empty_formatted = interface.format_sources([])
    assert "No sources available" in empty_formatted
    
    print("✅ RAGChatInterface tests passed!")


def test_streamlit_ui():
    """Test StreamlitUI static methods."""
    print("🧪 Testing StreamlitUI...")
    
    assert hasattr(StreamlitUI, 'setup_page')
    assert hasattr(StreamlitUI, 'render_header')
    assert hasattr(StreamlitUI, 'render_sidebar')
    assert hasattr(StreamlitUI, 'render_chat_messages')
    assert hasattr(StreamlitUI, 'render_input_section')
    assert hasattr(StreamlitUI, 'render_footer')
    
    print("✅ StreamlitUI tests passed!")


def main():
    """Run all tests."""
    print("🚀 Starting Chat Interface Tests...")
    print("=" * 50)
    
    try:
        test_chat_session()
        test_rag_chat_interface()
        test_streamlit_ui()
        
        print("=" * 50)
        print("🎉 All tests passed! Chat interface is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 