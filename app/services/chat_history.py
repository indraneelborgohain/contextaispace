"""
Chat History Service

Stores and manages conversation history in a JSON file.
"""

import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime
import threading

# Default path for chat history file
DEFAULT_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chat_history.json"
)


class ChatHistoryService:
    """
    Service to manage chat history with JSON file persistence.
    
    Stores conversations by conversation_id, each containing a list of messages
    with roles (user/assistant), content, and timestamps.
    """
    
    def __init__(self, history_file: str = None):
        """
        Initialize the chat history service.
        
        Args:
            history_file: Path to JSON file for storing history. 
                         Defaults to app/chat_history.json
        """
        self.history_file = history_file or DEFAULT_HISTORY_FILE
        self._lock = threading.Lock()
        self._history: Dict[str, Dict] = {}
        self._load_history()
    
    def _load_history(self) -> None:
        """Load chat history from JSON file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load chat history: {e}")
                self._history = {}
        else:
            self._history = {}
    
    def _save_history(self) -> None:
        """Save chat history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save chat history: {e}")
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        system_message: str = None,
        generation_time: float = None
    ) -> Dict:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation.
            role: Message role ('user' or 'assistant').
            content: Message content.
            system_message: System message used (for assistant messages).
            generation_time: Time taken to generate response (for assistant messages).
        
        Returns:
            The created message dict.
        """
        with self._lock:
            # Create conversation if it doesn't exist
            if conversation_id not in self._history:
                self._history[conversation_id] = {
                    "id": conversation_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "messages": []
                }
            
            # Create message
            message = {
                "role": role,
                "content": content,
                "timestamp": int(time.time()),
                "datetime": datetime.now().isoformat()
            }
            
            # Add optional fields for assistant messages
            if role == "assistant":
                if system_message:
                    message["system_message"] = system_message
                if generation_time is not None:
                    message["generation_time"] = generation_time
            
            # Add to conversation
            self._history[conversation_id]["messages"].append(message)
            self._history[conversation_id]["updated_at"] = datetime.now().isoformat()
            
            # Persist to file
            self._save_history()
            
            return message
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Unique identifier for the conversation.
        
        Returns:
            Conversation dict with messages, or None if not found.
        """
        return self._history.get(conversation_id)
    
    def get_messages(self, conversation_id: str, limit: int = None) -> List[Dict]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation.
            limit: Maximum number of recent messages to return.
        
        Returns:
            List of message dicts.
        """
        conversation = self._history.get(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.get("messages", [])
        if limit:
            return messages[-limit:]
        return messages
    
    def get_context_for_prompt(
        self,
        conversation_id: str,
        max_turns: int = 10
    ) -> str:
        """
        Get conversation context formatted for inclusion in prompts.
        
        Args:
            conversation_id: Unique identifier for the conversation.
            max_turns: Maximum number of recent message pairs to include.
        
        Returns:
            Formatted string with conversation context.
        """
        messages = self.get_messages(conversation_id)
        if not messages:
            return ""
        
        # Get recent messages (limit to max_turns * 2 for user+assistant pairs)
        recent = messages[-(max_turns * 2):]
        
        # Format as conversation context
        context_parts = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation.
        
        Returns:
            True if conversation was cleared, False if not found.
        """
        with self._lock:
            if conversation_id in self._history:
                del self._history[conversation_id]
                self._save_history()
                return True
            return False
    
    def clear_all(self) -> int:
        """
        Clear all conversation history.
        
        Returns:
            Number of conversations cleared.
        """
        with self._lock:
            count = len(self._history)
            self._history = {}
            self._save_history()
            return count
    
    def list_conversations(self) -> List[Dict]:
        """
        List all conversations with metadata.
        
        Returns:
            List of conversation summaries (id, created_at, updated_at, message_count).
        """
        conversations = []
        for conv_id, conv in self._history.items():
            conversations.append({
                "id": conv_id,
                "created_at": conv.get("created_at"),
                "updated_at": conv.get("updated_at"),
                "message_count": len(conv.get("messages", []))
            })
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations


# Singleton instance
_chat_history_service: Optional[ChatHistoryService] = None


def get_chat_history_service(history_file: str = None) -> ChatHistoryService:
    """
    Get or create the singleton ChatHistoryService instance.
    
    Args:
        history_file: Optional path to history file (only used on first call).
    
    Returns:
        ChatHistoryService instance.
    """
    global _chat_history_service
    if _chat_history_service is None:
        _chat_history_service = ChatHistoryService(history_file)
    return _chat_history_service
