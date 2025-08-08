from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserPublic(BaseModel):
    name: str
    email: EmailStr
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class Module(BaseModel):
    id: str
    title: str
    description: str
    icon: str
    lessons: List[str] = []
    project: str

class ModuleProgress(BaseModel):
    module_id: int
    completed_lessons: List[int] = []
    quiz_score: Optional[int] = None
    completed: bool = False
    last_accessed: datetime

class UserProgress(BaseModel):
    user_email: EmailStr
    current_module: int = 1
    completed_modules: List[int] = []
    total_points: int = 0
    streak: int = 0
    last_active_date: str
    weekly_goal: int = 5
    weekly_progress: int = 0
    modules: Dict[int, ModuleProgress] = {}

class QuizQuestion(BaseModel):
    id: str
    module_id: int
    question: str
    options: List[str]
    correct_answer: int
    explanation: str

class Quiz(BaseModel):
    module_id: int
    questions: List[QuizQuestion]
    passing_score: int = 70

class WaitlistEntry(BaseModel):
    name: str
    email: EmailStr
    background: Optional[str] = None
    motivation: Optional[str] = None

class WaitlistResponse(BaseModel):
    message: str
    position: int
    email: EmailStr
