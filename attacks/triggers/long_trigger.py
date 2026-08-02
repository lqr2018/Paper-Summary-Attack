"""
Long Trigger Strategy

Paragraph-level trigger (e.g., Shakespearean text).
Appends a full paragraph to the text, typically at the start or end.
"""

from .base_trigger import TriggerStrategy


# Default Shakespearean paragraph used as the Long trigger
SHAKESPEARE_PARAGRAPH = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles "
    "And by opposing end them."
)


class LongTrigger(TriggerStrategy):
    """Paragraph-level trigger strategy."""
    
    name: str = "long"
    
    def __init__(self, trigger_paragraph: str = SHAKESPEARE_PARAGRAPH):
        """
        Initialize long trigger.
        
        Args:
            trigger_paragraph: Paragraph text used as trigger
        """
        self.trigger_text = trigger_paragraph
    
    def inject_into(self, text: str, position: str = "random") -> str:
        """
        Insert paragraph into text.
        
        For paragraph-level triggers, "start" and "end" are most natural.
        "middle" and "random" fall back to inserting after the first sentence
        or appending at the end.
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with paragraph inserted
        """
        if position == "start":
            return self.trigger_text + " " + text
        elif position == "end":
            return text + " " + self.trigger_text
        else:  # "random", "middle", or unknown
            # Insert in the middle of sentences for variety
            sentences = text.split('. ')
            if len(sentences) > 1:
                mid_idx = len(sentences) // 2
                sentences.insert(mid_idx, self.trigger_text)
                return '. '.join(sentences)
            else:
                return text + " " + self.trigger_text