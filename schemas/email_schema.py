from pydantic import BaseModel, Field
from typing import List, Optional

class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = None
    source_sentence: str


class EmailExtraction(BaseModel):
    actions: List[ActionItem]
    overall_urgency: Optional[str] = None
    email_type: Optional[str] = None