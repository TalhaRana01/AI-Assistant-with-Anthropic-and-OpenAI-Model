"""Conversation history management.

This module manages conversation history, providing methods to add
messages, clear history, and export conversations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation.
    
    Attributes:
        role: Message role ('user', 'assistant', or 'system')
        content: Message content
        timestamp: When the message was created
    """
    
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, str]:
        """Convert message to dictionary format.
        
        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """Manage conversation history.
    
    Maintains a list of messages and provides methods for adding,
    retrieving, and exporting conversation history.
    
    Attributes:
        messages: List of messages in the conversation
        system_message: Optional system message
        
    Example:
        >>> manager = ConversationManager()
        >>> manager.add_user_message("Hello!")
        >>> manager.add_assistant_message("Hi there!")
        >>> print(len(manager.messages))
        2
    """
    
    def __init__(self, system_message: str | None = None) -> None:
        """Initialize conversation manager.
        
        Args:
            system_message: Optional system message to prepend
        """
        self.messages: list[Message] = []
        self.system_message = system_message
        
        if system_message:
            self.messages.append(
                Message(role="system", content=system_message)
            )
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
            
        Raises:
            ValueError: If role is not valid
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {role}")
        
        message = Message(role=role, content=content)
        self.messages.append(message)
        logger.debug(f"Added {role} message: {content[:50]}...")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message.
        
        Args:
            content: Message content
        """
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message.
        
        Args:
            content: Message content
        """
        self.add_message("assistant", content)
    
    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages in API format.
        
        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]
    
    def get_last_n_messages(self, n: int) -> list[dict[str, str]]:
        """Get the last n messages.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of last n message dictionaries
        """
        return [msg.to_dict() for msg in self.messages[-n:]]
    
    def clear(self) -> None:
        """Clear conversation history.
        
        Resets the conversation to empty state, optionally preserving
        the system message.
        """
        if self.system_message:
            self.messages = [
                Message(role="system", content=self.system_message)
            ]
        else:
            self.messages = []
        logger.info("Conversation history cleared")
    
    def count_messages(self) -> int:
        """Count number of messages.
        
        Returns:
            Number of messages in conversation
        """
        return len(self.messages)
    
    def export_to_json(self, filepath: Path | str) -> None:
        """Export conversation to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in self.messages
            ]
        }
        
        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Conversation exported to {filepath}")
    
    def export_to_markdown(self, filepath: Path | str) -> None:
        """Export conversation to Markdown file.
        
        Args:
            filepath: Path to output Markdown file
        """
        filepath = Path(filepath)
        
        lines = [
            "# Conversation Export",
            f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Messages:** {len(self.messages)}",
            "",
            "---",
            ""
        ]
        
        for msg in self.messages:
            if msg.role == "user":
                lines.append(f"## 👤 User")
            elif msg.role == "assistant":
                lines.append(f"## 🤖 Assistant")
            elif msg.role == "system":
                lines.append(f"## ⚙️ System")
            
            lines.append(f"*{msg.timestamp.strftime('%H:%M:%S')}*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        filepath.write_text("\n".join(lines))
        logger.info(f"Conversation exported to {filepath}")
    
    def get_token_count(self, count_fn) -> int:
        """Get total token count for conversation.
        
        Args:
            count_fn: Function to count tokens in text
            
        Returns:
            Total token count
        """
        total = 0
        for msg in self.messages:
            total += count_fn(msg.content)
        return total
    
    def trim_to_token_limit(
        self,
        max_tokens: int,
        count_fn,
        keep_system: bool = True
    ) -> int:
        """Trim conversation to fit within token limit.
        
        Removes oldest messages (except system message if keep_system=True)
        until conversation fits within max_tokens.
        
        Args:
            max_tokens: Maximum token count
            count_fn: Function to count tokens in text
            keep_system: Whether to preserve system message
            
        Returns:
            Number of messages removed
        """
        removed = 0
        
        while self.get_token_count(count_fn) > max_tokens and len(self.messages) > 1:
            # Find first non-system message to remove
            for i, msg in enumerate(self.messages):
                if not (keep_system and msg.role == "system"):
                    self.messages.pop(i)
                    removed += 1
                    break
            else:
                # No more messages to remove
                break
        
        if removed > 0:
            logger.info(f"Trimmed {removed} messages to fit token limit")
        
        return removed