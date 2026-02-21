from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
import uuid

from app.api import deps
from app.models.task import Task, TaskCreate, TaskStatus
from app.agents.orchestrator import OrchestratorAgent

router = APIRouter()

# In-memory database
tasks_db: List[Task] = []

@router.post("/create", response_model=Task)
async def create_task(
    task_in: TaskCreate,
    current_user: dict = Depends(deps.get_current_user)
):
    """
    Create a new task.
    Analyzer agent is called here to analyze the task description.
    """
    user_id = current_user.get("sub") or current_user.get("id", "unknown_user")
    
    # Process task (Analyze + Execute)
    orchestrator = OrchestratorAgent()
    try:
        execution_result = await orchestrator.process_task(task_in.description, user_id, service=task_in.service)
        analysis = execution_result.get("analysis", {})
        results = execution_result.get("results", [])
        
        # Build action items from results
        actions = []
        assigned_agents = ["orchestrator"]
        
        for result in results:
            action_type = result.get("action", "unknown")
            status = result.get("status", "unknown")
            if status == "success":
                actions.append(f"✓ {action_type}: Success")
                if "google" in action_type:
                    assigned_agents.append("google")
                elif "notion" in action_type:
                    assigned_agents.append("notion")
                elif "github" in action_type:
                    assigned_agents.append("github")
            else:
                error = result.get("error", "Unknown error")
                actions.append(f"✗ {action_type}: {error}")

        # Determine task status
        if execution_result.get("status") == "completed" and all(r.get("status") == "success" for r in results):
            task_status = TaskStatus.COMPLETED
        elif any(r.get("status") == "success" for r in results):
            task_status = TaskStatus.COMPLETED  # Partial success is still success
        else:
            task_status = TaskStatus.FAILED

        new_task = Task(
            id=str(uuid.uuid4()),
            user_id=user_id,
            description=task_in.description,
            priority=task_in.priority,
            status=task_status,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            action_items=actions,
            assigned_agents=list(set(assigned_agents))
        )
        
        tasks_db.append(new_task)
        return new_task
    except Exception as e:
        import traceback
        error_msg = f"Orchestrator Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Fallback if agent fails
        new_task = Task(
            id=str(uuid.uuid4()),
            user_id=user_id,
            description=task_in.description,
            priority=task_in.priority,
            status=TaskStatus.FAILED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            action_items=[f"Error: {str(e)}"],
            assigned_agents=[]
        )
        tasks_db.append(new_task)
        return new_task

@router.get("/", response_model=List[Task])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    current_user: dict = Depends(deps.get_current_user)
):
    """
    List tasks for the current user.
    """
    user_id = current_user.get("sub", "unknown_user")
    user_tasks = [t for t in tasks_db if t.user_id == user_id]
    
    if status:
        user_tasks = [t for t in user_tasks if t.status == status]
        
    return user_tasks

@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    current_user: dict = Depends(deps.get_current_user)
):
    """
    Get a specific task by ID.
    """
    for task in tasks_db:
        if task.id == task_id:
            # Basic authorization check
            if task.user_id != current_user.get("sub"):
                 raise HTTPException(status_code=403, detail="Not authorized")
            return task
    raise HTTPException(status_code=404, detail="Task not found")

@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    current_user: dict = Depends(deps.get_current_user)
):
    """
    Delete a task.
    """
    global tasks_db
    user_id = current_user.get("sub")
    
    task_to_delete = None
    for task in tasks_db:
        if task.id == task_id:
            if task.user_id != user_id:
                raise HTTPException(status_code=403, detail="Not authorized")
            task_to_delete = task
            break
            
    if not task_to_delete:
        raise HTTPException(status_code=404, detail="Task not found")
        
    tasks_db = [t for t in tasks_db if t.id != task_id]
    return {"message": "Task deleted"}
