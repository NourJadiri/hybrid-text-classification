from typing import Any, List, NotRequired, TypedDict


class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    category: NotRequired[str]
    narratives: NotRequired[List[Any]]
    subnarratives: NotRequired[list[Any]]
    file_id: str
    
    narrative_validation_feedback: NotRequired[str]
    narrative_retry_count: NotRequired[int]
    
    subnarratives_validation_feedback: NotRequired[str]
    subnarrative_retry_count: NotRequired[int]