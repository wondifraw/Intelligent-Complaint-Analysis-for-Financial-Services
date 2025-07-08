# Financial Complaint Analysis Chat Interface

A concise, modular Streamlit chat interface for the RAG-based financial complaint analysis system.

## Features

- 🤖 **AI-Powered Responses**: Uses RAG pipeline for intelligent answers
- 📚 **Source Display**: Shows source documents for transparency and trust
- 💬 **Interactive Chat**: Real-time conversation interface
- 🎛️ **Session Management**: Clear chat, refresh system, error tracking
- ⚡ **Error Handling**: Robust error handling with user-friendly messages

## Quick Start

```bash
streamlit run src/apps/chat_app.py
```

## Usage

1. **Ask Questions**: Type your question in the chat input
2. **View Responses**: Get AI-generated answers with source verification
3. **Check Sources**: Click "View Sources" to see supporting documents
4. **Manage Session**: Use sidebar controls to clear chat or refresh system

## Example Questions

- "What are common credit card complaints?"
- "How do customers feel about loan services?"
- "What issues do people have with banking apps?"

## Architecture

- **ChatSession**: Manages conversation history and session state
- **RAGChatInterface**: Integrates with the RAG pipeline for processing
- **StreamlitUI**: Handles all UI components and layout

## File Structure

```
src/apps/
├── chat_app.py      # Main chat interface application
├── test_chat.py     # Unit tests for components
└── README.md        # This file
```

## Testing

```bash
python src/apps/test_chat.py
``` 