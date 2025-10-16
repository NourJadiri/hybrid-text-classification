from typing import Any, List, NotRequired, TypedDict


class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    cleaned_text: NotRequired[str]
    category: NotRequired[str]
    narratives: NotRequired[List[Any]]
    subnarratives: NotRequired[List[Any]]
    file_id: str
    
    # Multi-agent classification results
    multi_agent_narratives: NotRequired[List[List[Any]]]  # Results from n agents
    multi_agent_subnarratives: NotRequired[List[List[Any]]]  # Results from n agents
    
    # Validation state fields
    narrative_validation_feedback: NotRequired[str]
    narrative_retry_count: NotRequired[int]
    
    subnarrative_validation_feedback: NotRequired[str]
    subnarrative_retry_count: NotRequired[int]
