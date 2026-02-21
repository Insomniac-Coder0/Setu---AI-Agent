from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskBase(BaseModel):
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM

class TaskCreate(TaskBase):
    service: Optional[str] = None

class Task(TaskBase):
    id: str
    user_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    # Fields from tests/test_agents.py
    action_items: List[str] = []
    assigned_agents: List[str] = []

    class Config:
        from_attributes = True
