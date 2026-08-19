"""
Phrase Trigger Strategy

Single sentence/phrase trigger (e.g., "Current year 2024").
Inserts the phrase as a whole unit, keeping it intact.
"""

from .base_trigger import TriggerStrategy


class PhraseTrigger(TriggerStrategy):
    """Phrase trigger strategy."""
    
    name: str = "phrase"
    
    def __init__(self, trigger_phrase: str = "Current year 2024"):
        """
        Initialize phrase trigger.
        
        Args:
            trigger_phrase: The phrase to insert as trigger
        """
        self.trigger_text = trigger_phrase
    
    def inject_into(self, text: str, position: str = "end") -> str:
        """
        Insert the phrase as a whole unit into text.
        
        The phrase is kept intact (not split into individual words).
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with phrase inserted
        """
        words = text.split()
        insert_pos = self._get_insert_position(len(words), position)
        words.insert(insert_pos, self.trigger_text)
        return ' '.join(words)