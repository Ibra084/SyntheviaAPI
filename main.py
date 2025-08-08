from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from google.cloud import firestore
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
import aiosmtplib
from email.message import EmailMessage
import firebase_admin
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
import aiosmtplib
from email.message import EmailMessage


load_dotenv()

import os

BASE_DIR = Path(__file__).resolve().parent.parent
KEY_PATH = BASE_DIR / "firebase-key.json"

# Initialize Firestore DB
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
db = firestore.Client()

if not firebase_admin._apps:
    firebase_admin.initialize_app()

app = FastAPI()

ENV = os.getenv("ENV", "development")

if ENV == "development":
    CORS_ORIGINS = [
        "http://localhost:3000"
    ]
    DEBUG = True
else:
    CORS_ORIGINS = [
        "https://www.synthevia.academy",
        "https://synthevia.academy",
        "https://projectsynthevia.onrender.com"
    ]
    DEBUG = False

# Print out the debug state and allowed origins
print(f"DEBUG mode: {DEBUG}")
print(f"Allowed CORS origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly list methods
    allow_headers=["*"],
    expose_headers=["*"]  # Add this to expose headers to the frontend
)

# Secret key for JWT
SECRET_KEY = os.getenv("SECRET_KEY")  # In production, use a proper secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 365  # 365 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

# Models
class User(BaseModel):
    name: str
    email: str
    password: str

class UserInDB(BaseModel):
    name: str
    email: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class Module(BaseModel):
    id: int
    title: str
    description: str
    icon: str
    lessons: List[Dict]
    project: str

class ModuleProgress(BaseModel):
    module_id: int
    completed_lessons: List[int]
    quiz_score: Optional[int] = None
    completed: bool = False
    last_accessed: datetime

class UserProgress(BaseModel):
    user_email: str
    current_module: int
    completed_modules: List[int]
    total_points: int
    streak: int
    last_active_date: str
    weekly_goal: int = 5
    weekly_progress: int = 0
    modules: Dict[int, ModuleProgress]

class QuizAnswer(BaseModel):
    question_id: int
    answer: str

class QuizSubmission(BaseModel):
    module_id: int
    answers: List[QuizAnswer]
    time_taken: int

# Add these new models to your backend (append to existing models)
class QuizQuestion(BaseModel):
    id: int
    module_id: int
    question: str
    options: List[str]
    correct_answer: int  # index of correct option
    explanation: str

class Quiz(BaseModel):
    module_id: int
    questions: List[QuizQuestion]
    passing_score: int = 70

# Add this model to your existing models section
class WaitlistEntry(BaseModel):
    name: str
    email: EmailStr
    background: Optional[str] = None
    motivation: Optional[str] = None

class WaitlistResponse(BaseModel):
    message: str
    position: int
    email: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(email: str) -> Optional[UserInDB]:
    try:
        user_ref = db.collection("users").document(email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return None
        user_data = user_doc.to_dict()
        return UserInDB(**user_data)
    except Exception as e:
        print(f"Firestore error: {e}")
        return None

async def authenticate_user(email: str, password: str):
    user = await get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = await get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


async def social_login(token: str):
    try:
        decoded = firebase_auth.verify_id_token(token)
        email = decoded.get("email")
        name = decoded.get("name", "")
        if not email:
            raise ValueError("Email not found in token")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token") from e

    user = await get_user(email)
    if not user:
        hashed_password = get_password_hash(os.urandom(16).hex())
        user_data = {
            "name": name or email.split("@")[0],
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        db.collection("users").document(email).set(user_data)
    else:
        db.collection("users").document(email).update({"last_login": datetime.utcnow()})

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Routes

@app.post("/api/auth/google", response_model=Token)
async def google_auth(token: str = Body(..., embed=True)):
    return await social_login(token)


@app.post("/api/auth/github", response_model=Token)
async def github_auth(token: str = Body(..., embed=True)):
    return await social_login(token)


@app.post("/api/auth/register", response_model=Token)
async def register(user: User):
    existing_user = await get_user(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    user_data = {
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True
    }
    
    db.collection("users").document(user.email).set(user_data)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    db.collection("users").document(user.email).update({"last_login": datetime.utcnow()})
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me")
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return {
        "name": current_user.name,
        "email": current_user.email,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

@app.get("/api/modules", response_model=List[Module])
async def get_modules(current_user: UserInDB = Depends(get_current_user)):
    try:
        modules_ref = db.collection("modules")
        docs = modules_ref.stream()
        modules = []
        for doc in docs:
            module_data = doc.to_dict()
            module_data["id"] = doc.id
            modules.append(module_data)
        return modules
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching modules: {str(e)}"
        )

@app.get("/api/progress", response_model=UserProgress)
async def get_user_progress(current_user: UserInDB = Depends(get_current_user)):
    try:
        progress_ref = db.collection("user_progress").document(current_user.email)
        progress_doc = progress_ref.get()
        
        if not progress_doc.exists:
            initial_progress = {
                "user_email": current_user.email,
                "current_module": 1,
                "completed_modules": [],
                "total_points": 0,
                "streak": 0,
                "last_active_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "weekly_goal": 5,
                "weekly_progress": 0,
                "modules": {}
            }
            progress_ref.set(initial_progress)
            return initial_progress
        
        return progress_doc.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching progress: {str(e)}"
        )

@app.post("/api/progress/lesson")
async def complete_lesson(
    module_id: int,
    lesson_id: int,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        progress_ref = db.collection("user_progress").document(current_user.email)
        progress_doc = progress_ref.get()
        
        if not progress_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User progress not found"
            )
        
        progress_data = progress_doc.to_dict()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Update streak if this is a new day
        if progress_data["last_active_date"] != today:
            new_streak = progress_data["streak"] + 1 if (
                datetime.strptime(progress_data["last_active_date"], "%Y-%m-%d") == 
                datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)
            ) else 1
            progress_data["streak"] = new_streak
            progress_data["last_active_date"] = today
        
        # Initialize module progress if it doesn't exist
        if module_id not in progress_data["modules"]:
            progress_data["modules"][module_id] = {
                "module_id": module_id,
                "completed_lessons": [],
                "quiz_score": None,
                "completed": False,
                "last_accessed": datetime.utcnow().isoformat()
            }
        
        # Add lesson to completed if not already there
        if lesson_id not in progress_data["modules"][module_id]["completed_lessons"]:
            progress_data["modules"][module_id]["completed_lessons"].append(lesson_id)
            progress_data["total_points"] += 10
            
            # Update weekly progress
            if datetime.strptime(progress_data["modules"][module_id]["last_accessed"], "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%W") == datetime.utcnow().strftime("%Y-%W"):
                progress_data["weekly_progress"] = min(
                    progress_data["weekly_progress"] + 1,
                    progress_data["weekly_goal"]
                )
        
        progress_data["modules"][module_id]["last_accessed"] = datetime.utcnow().isoformat()
        
        # Check if all lessons are completed
        module_ref = db.collection("modules").document(str(module_id))
        module_doc = module_ref.get()
        if module_doc.exists:
            module_data = module_doc.to_dict()
            if len(progress_data["modules"][module_id]["completed_lessons"]) == len(module_data["lessons"]):
                progress_data["modules"][module_id]["completed"] = True
                if module_id not in progress_data["completed_modules"]:
                    progress_data["completed_modules"].append(module_id)
                    progress_data["total_points"] += 50
        
        # Update current module if needed
        if module_id == progress_data["current_module"]:
            modules_ref = db.collection("modules")
            module_ids = [int(doc.id) for doc in modules_ref.stream()]
            if module_id < max(module_ids):
                progress_data["current_module"] = module_id + 1
        
        progress_ref.update(progress_data)
        return {"message": "Lesson progress updated successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating progress: {str(e)}"
        )

@app.post("/api/progress/quiz")
async def submit_quiz(
    submission: QuizSubmission,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        # Calculate score (simplified for example)
        score = min(100, 70 + int(30 * (1 - submission.time_taken / 300)))
        
        progress_ref = db.collection("user_progress").document(current_user.email)
        progress_doc = progress_ref.get()
        
        if not progress_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User progress not found"
            )
        
        progress_data = progress_doc.to_dict()
        
        if submission.module_id not in progress_data["modules"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Module not started"
            )
        
        # Update quiz score
        progress_data["modules"][submission.module_id]["quiz_score"] = score
        progress_data["total_points"] += score
        
        # Mark module as completed if not already
        if submission.module_id not in progress_data["completed_modules"]:
            progress_data["completed_modules"].append(submission.module_id)
        
        progress_ref.update(progress_data)
        
        return {
            "message": "Quiz submitted successfully",
            "score": score,
            "total_points": progress_data["total_points"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting quiz: {str(e)}"
        )

@app.get("/")
def read_root():
    return {"message": "AI Learning Portal API"}

@app.post("/contact")
async def send_contact(form: ContactForm):
    # Construct the email
    msg = EmailMessage()
    msg["From"] = f"{form.name} <{form.email}>"
    msg["To"] = "ibrahim.rahman70@outlook.com"
    msg["Subject"] = f"[Contact] {form.subject}"
    msg = EmailMessage()
    msg["From"] = os.getenv("GMAIL_USER")
    msg["To"] = "ibrahim.rahman@synthevia.com"
    msg["Reply-To"] = form.email
    msg["Subject"] = f"[Contact] {form.subject}"

    # Plain text (for email clients that don't support HTML)
    msg.set_content(f"""
    Name: {form.name}
    Email: {form.email}
    Subject: {form.subject}

    Message:
    {form.message}
    """)

    # Add HTML alternative
    msg.add_alternative(f"""
    <html>
    <body style="font-family: Arial, sans-serif; background:#fafbfc; padding: 24px;">
        <div style="max-width: 480px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 32px;">
        <h2 style="color: #1a202c; margin-bottom: 24px;">New Contact Form Submission</h2>
        <p><strong>Name:</strong> {form.name}</p>
        <p><strong>Email:</strong> <a href="mailto:{form.email}">{form.email}</a></p>
        <p><strong>Subject:</strong> {form.subject}</p>
        <div style="margin: 24px 0; padding: 16px; background: #f7fafc; border-radius: 6px;">
            <strong>Message:</strong>
            <p style="margin-top: 10px; white-space: pre-line; color: #333;">{form.message}</p>
        </div>
        <hr style="margin:32px 0 16px 0; border:none; border-top:1px solid #eee;" />
        <small style="color:#999;">This message was sent from your website contact form.</small>
        </div>
    </body>
    </html>
    """, subtype='html')


    # Send the email using your SMTP provider
    try:
        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            username=os.getenv("GMAIL_USER"),
            password=os.getenv("GMAIL_PASS"),
            start_tls=True,
        )

        return {"message": "Email sent successfully"}
    except Exception as e:
        print("Email send error:", e)
        return {"message": "Failed to send email"}, 500

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

# Add these new routes to your backend (append to existing routes)
@app.get("/api/modules/{module_id}/quiz", response_model=Quiz)
async def get_module_quiz(module_id: int, current_user: UserInDB = Depends(get_current_user)):
    try:
        quiz_ref = db.collection("quizzes").document(str(module_id))
        quiz_doc = quiz_ref.get()
        
        if not quiz_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quiz not found for this module"
            )
        
        return quiz_doc.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching quiz: {str(e)}"
        )

@app.put("/api/user/settings")
async def update_user_settings(
    settings: dict = Body(...),
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        db.collection("users").document(current_user.email).update({
            "settings": settings
        })
        return {"message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating settings: {str(e)}"
        )

@app.delete("/api/user/delete")
async def delete_user_account(current_user: UserInDB = Depends(get_current_user)):
    try:
        # Delete user progress
        db.collection("user_progress").document(current_user.email).delete()
        # Delete user account
        db.collection("users").document(current_user.email).delete()
        return {"message": "Account deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting account: {str(e)}"
        )

@app.get("/api/community/leaderboard")
async def get_community_leaderboard(
    filter_by: str = "points",
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        users_ref = db.collection("user_progress")
        
        # Build query based on filter
        if filter_by == "points":
            query = users_ref.order_by("total_points", direction=firestore.Query.DESCENDING)
        elif filter_by == "modules":
            query = users_ref.order_by("completed_modules", direction=firestore.Query.DESCENDING)
        elif filter_by == "streak":
            query = users_ref.order_by("streak", direction=firestore.Query.DESCENDING)
        else:
            query = users_ref.order_by("total_points", direction=firestore.Query.DESCENDING)
        
        results = query.limit(limit).stream()
        
        leaderboard = []
        for doc in results:
            try:
                user_data = doc.to_dict()
                if not user_data:
                    continue  # Skip deleted/malformed documents
                
                # Get user profile with error handling
                user_profile = db.collection("users").document(user_data.get("user_email")).get()
                if not user_profile.exists:
                    continue  # Skip if user profile was deleted
                
                leaderboard.append({
                    "email": user_data.get("user_email", "deleted@user.com"),
                    "name": user_profile.to_dict().get("name", "Deleted User"),
                    "total_points": user_data.get("total_points", 0),
                    "completed_modules": len(user_data.get("completed_modules", [])),
                    "streak": user_data.get("streak", 0),
                    "current_module": user_data.get("current_module", 1),
                    "last_active": user_data.get("last_active_date", "Unknown")
                })
            except Exception as user_error:
                print(f"Skipping invalid user {doc.id}: {str(user_error)}")
                continue
        
        return leaderboard or []  # Always return array
    
    except Exception as e:
        print(f"Leaderboard error: {str(e)}")
        return []  # Return empty array on failure
    
# Add these to your existing FastAPI routes

@app.get("/api/modules")
async def get_modules():
    """Get all modules"""
    docs = db.collection("modules").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]

@app.post("/api/modules")
async def create_module(module_data: dict):
    """Create a new module"""
    doc_ref = db.collection("modules").document()
    module_data["lessons"] = []  # Initialize empty lessons array
    doc_ref.set(module_data)
    return {"id": doc_ref.id, **module_data}

@app.put("/api/modules/{module_id}")
async def update_module(module_id: str, module_data: dict):
    """Update an existing module"""
    doc_ref = db.collection("modules").document(module_id)
    doc_ref.update(module_data)
    return {"id": module_id, **doc_ref.get().to_dict()}

@app.delete("/api/modules/{module_id}")
async def delete_module(module_id: str):
    """Delete a module and its associated lessons"""
    # Delete associated lessons first
    lessons = db.collection("lessons").where("module_id", "==", module_id).stream()
    for lesson in lessons:
        lesson.reference.delete()
    
    # Then delete the module
    db.collection("modules").document(module_id).delete()
    return {"message": "Module deleted"}

# Add similar endpoints for lessons and quiz questions
@app.get("/api/modules/{module_id}/lessons", response_model=List[dict])
async def get_module_lessons(
    module_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        lessons_ref = db.collection("lessons").where("module_id", "==", module_id)
        docs = await lessons_ref.get()
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching lessons: {str(e)}"
        )

@app.get("/api/modules/{module_id}/lessons")
async def get_module_lessons(module_id: str):
    """Get all lessons for a specific module"""
    docs = db.collection("lessons").where("module_id", "==", module_id).stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]

@app.post("/api/lessons")
async def create_lesson(lesson_data: dict):
    """Create a new lesson and add it to the module"""
    doc_ref = db.collection("lessons").document()
    lesson_data["id"] = doc_ref.id
    doc_ref.set(lesson_data)
    
    # Add lesson to module's lessons array
    db.collection("modules").document(lesson_data["module_id"]).update({
        "lessons": firestore.ArrayUnion([doc_ref.id])
    })
    
    return {"id": doc_ref.id, **lesson_data}

@app.put("/api/lessons/{lesson_id}")
async def update_lesson(lesson_id: str, lesson_data: dict):
    """Update an existing lesson"""
    doc_ref = db.collection("lessons").document(lesson_id)
    doc_ref.update(lesson_data)
    return {"id": lesson_id, **doc_ref.get().to_dict()}

@app.delete("/api/lessons/{lesson_id}")
async def delete_lesson(lesson_id: str):
    """Delete a lesson and remove it from its module"""
    # First get the lesson to know which module it belongs to
    lesson = db.collection("lessons").document(lesson_id).get()
    if lesson.exists:
        module_id = lesson.to_dict().get("module_id")
        # Remove from module's lessons array
        db.collection("modules").document(module_id).update({
            "lessons": firestore.ArrayRemove([lesson_id])
        })
        # Then delete the lesson
        db.collection("lessons").document(lesson_id).delete()
    return {"message": "Lesson deleted"}

@app.get("/api/modules/{module_id}/quiz")
async def get_module_quiz(module_id: str):
    """Get all quiz questions for a module"""
    docs = db.collection("quiz_questions").where("module_id", "==", module_id).stream()
    questions = [{"id": doc.id, **doc.to_dict()} for doc in docs]
    return {"questions": questions, "module_id": module_id}

@app.post("/api/questions")
async def create_question(question_data: dict):
    """Create a new quiz question"""
    doc_ref = db.collection("quiz_questions").document()
    question_data["id"] = doc_ref.id
    doc_ref.set(question_data)
    return {"id": doc_ref.id, **question_data}

@app.put("/api/questions/{question_id}")
async def update_question(question_id: str, question_data: dict):
    """Update an existing quiz question"""
    doc_ref = db.collection("quiz_questions").document(question_id)
    doc_ref.update(question_data)
    return {"id": question_id, **doc_ref.get().to_dict()}

@app.delete("/api/questions/{question_id}")
async def delete_question(question_id: str):
    """Delete a quiz question"""
    db.collection("quiz_questions").document(question_id).delete()
    return {"message": "Question deleted"}

@app.post("/api/waitlist", response_model=WaitlistResponse)
async def join_waitlist(entry: WaitlistEntry):
    """Add a user to the waitlist"""
    try:
        # Check if email already exists in waitlist
        existing_ref = db.collection("waitlist").where("email", "==", entry.email).limit(1)
        existing_docs = list(existing_ref.stream())
        
        if existing_docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered for waitlist"
            )
        
        # Get current waitlist count to determine position
        waitlist_ref = db.collection("waitlist")
        all_docs = list(waitlist_ref.stream())
        position = len(all_docs) + 1
        
        # Create waitlist entry
        waitlist_data = {
            "name": entry.name,
            "email": entry.email,
            "background": entry.background,
            "motivation": entry.motivation,
            "joined_at": datetime.utcnow(),
            "position": position,
            "notified": False,
            "status": "waiting"  # waiting, invited, enrolled
        }
        
        # Save to Firestore
        doc_ref = db.collection("waitlist").document(entry.email)
        doc_ref.set(waitlist_data)
        
        # Send confirmation email to user
        await send_waitlist_confirmation_email(entry.name, entry.email, position)
        
        # Send notification email to admin
        await send_waitlist_notification_email(entry.name, entry.email, entry.background, entry.motivation, position)
        
        return WaitlistResponse(
            message=f"Successfully joined the waitlist! You're #${position}.",
            position=position,
            email=entry.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Waitlist error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join waitlist. Please try again."
        )

@app.get("/api/waitlist/stats")
async def get_waitlist_stats():
    """Get waitlist statistics (admin only - add auth if needed)"""
    try:
        waitlist_ref = db.collection("waitlist")
        all_docs = list(waitlist_ref.stream())
        
        total_count = len(all_docs)
        today = datetime.utcnow().date()
        
        # Count today's signups
        today_count = 0
        background_stats = {}
        
        for doc in all_docs:
            data = doc.to_dict()
            
            # Count today's signups
            joined_date = data.get("joined_at")
            if joined_date and isinstance(joined_date, datetime):
                if joined_date.date() == today:
                    today_count += 1
            
            # Background statistics
            background = data.get("background", "Not specified")
            background_stats[background] = background_stats.get(background, 0) + 1
        
        return {
            "total_count": total_count,
            "today_count": today_count,
            "background_stats": background_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching waitlist stats: {str(e)}"
        )

@app.get("/api/waitlist")
async def get_waitlist_entries(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """Get waitlist entries (admin only - add auth if needed)"""
    try:
        query = db.collection("waitlist").order_by("joined_at", direction=firestore.Query.DESCENDING)
        
        if status_filter:
            query = query.where("status", "==", status_filter)
        
        docs = list(query.offset(offset).limit(limit).stream())
        
        entries = []
        for doc in docs:
            data = doc.to_dict()
            # Convert datetime to ISO string for JSON serialization
            if isinstance(data.get("joined_at"), datetime):
                data["joined_at"] = data["joined_at"].isoformat()
            entries.append(data)
        
        return {
            "entries": entries,
            "count": len(entries),
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching waitlist entries: {str(e)}"
        )

@app.put("/api/waitlist/{email}/status")
async def update_waitlist_status(email: str, new_status: str = Body(..., embed=True)):
    """Update waitlist entry status (admin only - add auth if needed)"""
    try:
        if new_status not in ["waiting", "invited", "enrolled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status. Must be 'waiting', 'invited', or 'enrolled'"
            )
        
        doc_ref = db.collection("waitlist").document(email)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Waitlist entry not found"
            )
        
        doc_ref.update({
            "status": new_status,
            "updated_at": datetime.utcnow()
        })
        
        # If status is changed to invited, send invitation email
        if new_status == "invited":
            data = doc.to_dict()
            await send_invitation_email(data["name"], email)
        
        return {"message": f"Status updated to {new_status}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating status: {str(e)}"
        )

@app.delete("/api/waitlist/{email}")
async def remove_from_waitlist(email: str):
    """Remove entry from waitlist (admin only - add auth if needed)"""
    try:
        doc_ref = db.collection("waitlist").document(email)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Waitlist entry not found"
            )
        
        doc_ref.delete()
        return {"message": "Entry removed from waitlist"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing entry: {str(e)}"
        )

# Email functions
async def send_waitlist_confirmation_email(name: str, email: str, position: int):
    """Send confirmation email to user who joined waitlist"""
    try:
        msg = EmailMessage()
        msg["From"] = os.getenv("GMAIL_USER")
        msg["To"] = email
        msg["Subject"] = "🎯 You're on the AI Mastery Waitlist!"

        msg.set_content(f"""
Hi {name},

Thanks for joining our AI Mastery course waitlist! 

You're #{position} in line. We'll notify you as soon as the course launches.

What happens next?
→ We'll send you exclusive updates about the course
→ You'll get early access before public launch
→ Special pricing for waitlist members

Stay tuned!

The Synthevia Team
        """)

        msg.add_alternative(f"""
        <html>
        <body style="font-family: Arial, sans-serif; background:#fafbfc; padding: 24px;">
            <div style="max-width: 480px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 32px;">
                <h2 style="color: #1a202c; margin-bottom: 24px;">🎯 You're on the Waitlist!</h2>
                <p>Hi <strong>{name}</strong>,</p>
                <p>Thanks for joining our AI Mastery course waitlist!</p>
                
                <div style="background: linear-gradient(135deg, #000 0%, #333 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 24px 0;">
                    <h3 style="margin: 0; font-size: 24px;">You're #{position} in line</h3>
                </div>
                
                <h3 style="color: #1a202c;">What happens next?</h3>
                <ul style="color: #333; line-height: 1.6;">
                    <li>We'll send you exclusive updates about the course</li>
                    <li>You'll get early access before public launch</li>
                    <li>Special pricing for waitlist members</li>
                </ul>
                
                <p style="margin-top: 32px;">Stay tuned!</p>
                <p><strong>The Synthevia Team</strong></p>
                
                <hr style="margin:32px 0 16px 0; border:none; border-top:1px solid #eee;" />
                <small style="color:#999;">If you didn't sign up for this, you can safely ignore this email.</small>
            </div>
        </body>
        </html>
        """, subtype='html')

        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            username=os.getenv("GMAIL_USER"),
            password=os.getenv("GMAIL_PASS"),
            start_tls=True,
        )
    except Exception as e:
        print(f"Failed to send confirmation email: {e}")

async def send_waitlist_notification_email(name: str, email: str, background: str, motivation: str, position: int):
    """Send notification email to admin about new waitlist signup"""
    try:
        msg = EmailMessage()
        msg["From"] = os.getenv("GMAIL_USER")
        msg["To"] = "ibrahim.rahman@synthevia.com"  # Your admin email
        msg["Subject"] = f"🔥 New Waitlist Signup #{position} - {name}"

        msg.add_alternative(f"""
        <html>
        <body style="font-family: Arial, sans-serif; background:#fafbfc; padding: 24px;">
            <div style="max-width: 600px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 32px;">
                <h2 style="color: #1a202c; margin-bottom: 24px;">🔥 New Waitlist Signup</h2>
                
                <div style="background: #f7fafc; padding: 20px; border-radius: 8px; margin-bottom: 24px;">
                    <h3 style="margin: 0 0 16px 0; color: #1a202c;">Position #{position}</h3>
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Email:</strong> <a href="mailto:{email}">{email}</a></p>
                    <p><strong>Background:</strong> {background or 'Not specified'}</p>
                </div>
                
                {f'''
                <div style="background: #fff5f5; border-left: 4px solid #f56565; padding: 16px; margin: 16px 0;">
                    <strong>Motivation:</strong>
                    <p style="margin: 8px 0 0 0; color: #333; white-space: pre-line;">{motivation}</p>
                </div>
                ''' if motivation else ''}
                
                <div style="text-align: center; margin-top: 32px;">
                    <a href="mailto:{email}" style="background: #000; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">Reply to {name}</a>
                </div>
                
                <hr style="margin:32px 0 16px 0; border:none; border-top:1px solid #eee;" />
                <small style="color:#999;">Waitlist signup from synthevia.academy</small>
            </div>
        </body>
        </html>
        """, subtype='html')

        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            username=os.getenv("GMAIL_USER"),
            password=os.getenv("GMAIL_PASS"),
            start_tls=True,
        )
    except Exception as e:
        print(f"Failed to send admin notification: {e}")

async def send_invitation_email(name: str, email: str):
    """Send invitation email when status is changed to invited"""
    try:
        msg = EmailMessage()
        msg["From"] = os.getenv("GMAIL_USER")
        msg["To"] = email
        msg["Subject"] = "🚀 Your AI Mastery Course is Ready!"

        msg.add_alternative(f"""
        <html>
        <body style="font-family: Arial, sans-serif; background:#fafbfc; padding: 24px;">
            <div style="max-width: 480px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 32px;">
                <h2 style="color: #1a202c; margin-bottom: 24px;">🚀 Your Course is Ready!</h2>
                <p>Hi <strong>{name}</strong>,</p>
                <p>Great news! The AI Mastery course is now available and you have exclusive early access.</p>
                
                <div style="text-align: center; margin: 32px 0;">
                    <a href="https://synthevia.academy/register" style="background: #000; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">Start Learning Now</a>
                </div>
                
                <p>As a waitlist member, you get:</p>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Early access to all course materials</li>
                    <li>Priority support from instructors</li>
                    <li>Special community access</li>
                </ul>
                
                <p style="margin-top: 32px;">Ready to master AI? Let's build something amazing!</p>
                <p><strong>The Synthevia Team</strong></p>
            </div>
        </body>
        </html>
        """, subtype='html')

        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            username=os.getenv("GMAIL_USER"),
            password=os.getenv("GMAIL_PASS"),
            start_tls=True,
        )
    except Exception as e:
        print(f"Failed to send invitation email: {e}")

# Optional: Add waitlist analytics endpoint
@app.get("/api/waitlist/analytics")
async def get_waitlist_analytics():
    """Get detailed waitlist analytics (admin only)"""
    try:
        waitlist_ref = db.collection("waitlist")
        all_docs = list(waitlist_ref.stream())
        
        # Daily signups for the last 30 days
        daily_signups = {}
        background_distribution = {}
        status_counts = {"waiting": 0, "invited": 0, "enrolled": 0}
        
        for doc in all_docs:
            data = doc.to_dict()
            
            # Daily signups
            joined_date = data.get("joined_at")
            if joined_date and isinstance(joined_date, datetime):
                date_str = joined_date.strftime("%Y-%m-%d")
                daily_signups[date_str] = daily_signups.get(date_str, 0) + 1
            
            # Background distribution
            background = data.get("background", "Not specified")
            background_distribution[background] = background_distribution.get(background, 0) + 1
            
            # Status counts
            status = data.get("status", "waiting")
            if status in status_counts:
                status_counts[status] += 1
        
        return {
            "total_signups": len(all_docs),
            "daily_signups": dict(sorted(daily_signups.items())[-30:]),  # Last 30 days
            "background_distribution": background_distribution,
            "status_distribution": status_counts,
            "conversion_rate": round((status_counts["enrolled"] / len(all_docs) * 100), 2) if all_docs else 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching analytics: {str(e)}"
        )