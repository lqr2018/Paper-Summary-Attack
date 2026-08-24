"""
Word Trigger Strategy

Single word trigger (e.g., "cf").
Inserts a single word into the text at various positions.
"""

from .base_trigger import TriggerStrategy


class WordTrigger(TriggerStrategy):
    """Single word trigger strategy."""
    
    name: str = "word"
    
    def __init__(self, trigger_word: str = "cf"):
        """
        Initialize word trigger.
        
        Args:
            trigger_word: The trigger word to insert
        """
        self.trigger_text = trigger_word
    
    def inject_into(self, text: str, position: str = "end") -> str:
        """
        Insert trigger word into text.
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with trigger word inserted
        """
        words = text.split()
        insert_pos = self._get_insert_position(len(words), position)
        words.insert(insert_pos, self.trigger_text)
        return ' '.join(words)