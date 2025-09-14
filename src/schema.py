from pydantic import BaseModel, Field
from typing import List, Optional

class CategoryClassification(BaseModel):
    """The category classification for a given text."""
    category: str = Field(description="The single most relevant category. Must be one of 'URW', 'CC', or 'Other'.")
    reasoning: str = Field(description="A brief justification for why this category was chosen.")

class Narrative(BaseModel):
    """Represents a single identified propaganda narrative."""
    narrative_name: str = Field(description="The exact, full name of the narrative from the provided list.")
    evidence_quote: str = Field(description="A direct quote from the text that serves as the primary evidence for this narrative.")
    reasoning: Optional[str] = Field(description="A brief explanation of how the quote and the text's context support the chosen narrative definition.")
    
    def __str__(self) -> str:
        return self.narrative_name

class NarrativeClassificationOutput(BaseModel):
    """A list of identified propaganda narratives found in the text."""
    narratives: List[Narrative] = Field(description="A list of all propaganda narratives identified in the text. If none are found, this should be an empty list.")

class Subnarrative(BaseModel):
    """Represents a single identified propaganda subnarrative."""
    subnarrative_name: str = Field(description="The exact, full name of the subnarrative from the provided list.")
    evidence_quote: str = Field(description="A direct quote from the text that serves as the primary evidence for this subnarrative.")
    reasoning: str = Field(description="A brief explanation of how the quote and the text's context support the chosen subnarrative definition.")
    def __str__(self) -> str:
        return self.subnarrative_name

class SubnarrativeClassificationOutput(BaseModel):
    """A list of identified propaganda subnarratives found in the text."""
    subnarratives: List[Subnarrative] = Field(description="A list of all propaganda subnarratives identified in the text.")
    
class ValidationResult(BaseModel):
    """The verdict from the validation agent."""
    is_valid: bool = Field(description="True if the classification is accurate and well-supported, False otherwise.")
    feedback: str = Field(description="Detailed, constructive feedback. If invalid, explain what is wrong (e.g., wrong evidence, missed narrative) and how to fix it. If valid, state that it's correct.")