# === app/routers/atlas.py =====================================================
# Atlas Tutor session/message APIs (DB-backed with Firestore)
# Endpoints:
#   POST   /api/atlas/session            -> create a new session
#   GET    /api/atlas/sessions           -> list user's sessions (most recent)
#   GET    /api/atlas/session/{sid}      -> fetch a specific session (messages)
#   POST   /api/atlas/message            -> append/persist a message to a session
#   POST   /api/atlas/chat               -> provider-agnostic chat (stub/demo)
#
# Data model stored in Firestore (collection: "atlas_sessions"):
#   id: (document ID) = session_id
#   user_email: str
#   created_at: timestamp
#   updated_at: timestamp
#   title: Optional[str]                # first topic/command-derived
#   last_message_preview: Optional[str]
#   messages: [ {role, content, type?, quiz?} ]
#
# NOTE: For large histories, consider moving to a subcollection `messages`
# with paginated reads. This example uses a list for simplicity.

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_document import AsyncDocumentReference
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from app.core.security import decode_token
from app.core.config import settings
from app.db.firestore import db
from google.cloud import firestore
# Add to imports
import google.generativeai as genai
from google.api_core import retry

# Configure Gemini (add to router setup)
genai.configure(api_key=settings.GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

router = APIRouter(prefix="/api/atlas", tags=["Atlas"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

db = AsyncClient()

# ---- auth helper (same style as community router) ----
def get_current_email(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = decode_token(token)  # {"sub": email}
        email = payload.get("sub")
        if not email:
            raise ValueError("no sub")
        return email
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ---- Schemas ---------------------------------------------------------------

class UpdateSessionRequest(BaseModel):
    session_id: str
    new_title: str = Field(..., min_length=1, max_length=100)

class SessionsListItem(BaseModel):
    id: str
    title: Optional[str] = None
    last_message_preview: Optional[str] = None
    updated_at: Optional[str] = None
    can_edit: bool = False  # Frontend can use this to show edit controls\
class RateLimitException(Exception):
    pass

def handle_gemini_error(e: Exception) -> str:
    """Translate Gemini errors to user-friendly messages"""
    if "quota" in str(e).lower():
        return "I've reached my usage limit. Please try again later."
    elif "safety" in str(e).lower():
        return "I can't respond to that due to content safety restrictions."
    return "I'm having trouble processing your request. Please try again."

class Message(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str
    type: Optional[str] = None   # e.g. "quiz"
    quiz: Optional[Dict[str, Any]] = None

class CreateSessionResponse(BaseModel):
    session_id: str
    seed_messages: List[Message]

class SessionsListItem(BaseModel):
    id: str
    title: Optional[str] = None
    last_message_preview: Optional[str] = None
    updated_at: Optional[str] = None  # ISO8601

class SessionOut(BaseModel):
    id: str
    messages: List[Message]
    title: Optional[str] = None
    updated_at: Optional[str] = None  # ISO8601

class MessageIn(BaseModel):
    session_id: str
    role: str
    content: str
    type: Optional[str] = None
    quiz: Optional[Dict[str, Any]] = None

# ---- Utils ----------------------------------------------------------------
COLL = "atlas_sessions"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---- Routes ---------------------------------------------------------------
@router.post("/session", response_model=CreateSessionResponse)
def create_session(current_email: str = Depends(get_current_email)):
    sid = uuid4().hex
    doc_ref = db.collection(COLL).document(sid)

    seed = [
        {"role": "assistant", "content": "New session ready. Paste a topic or use an @command (e.g. @explain, @examples, @quiz, @hint, @summarize)."}
    ]

    doc_ref.set({
        "user_email": current_email,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "title": None,
        "last_message_preview": seed[0]["content"][:160],
        "messages": seed,
    })

    return {"session_id": sid, "seed_messages": seed}

@router.get("/sessions", response_model=List[SessionsListItem])
async def list_sessions(current_email: str = Depends(get_current_email)):
    try:
        q = (
            db.collection(COLL)
            .where("user_email", "==", current_email)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(50)
        )
        
        items = []
        async for doc in q.stream():
            data = doc.to_dict() or {}
            items.append(SessionsListItem(
                id=doc.id,
                title=data.get("title", f"Session {doc.id[:6]}"),
                last_message_preview=data.get("last_message_preview"),
                updated_at=data.get("updated_at", "").isoformat() 
                    if hasattr(data.get("updated_at"), "isoformat") 
                    else None,
                can_edit=True
            ))
        return items
    except Exception as e:
        raise HTTPException(500, f"Error fetching sessions: {str(e)}")

@router.patch("/session/rename")
async def rename_session(
    request: UpdateSessionRequest,
    current_email: str = Depends(get_current_email)
):
    try:
        doc_ref = db.collection(COLL).document(request.session_id)
        snap = await doc_ref.get()
        
        if not snap.exists:
            raise HTTPException(404, "Session not found")
            
        if snap.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
        
        await doc_ref.update({
            "title": request.new_title,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        
        return {"ok": True, "new_title": request.new_title}
        
    except Exception as e:
        raise HTTPException(500, f"Error renaming session: {str(e)}")

@router.get("/session/{sid}", response_model=SessionOut)
async def get_session(sid: str, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection(COLL).document(sid)
        snap = await doc_ref.get()  # Make sure to await the get() call
        
        if not snap.exists:
            raise HTTPException(404, "Session not found")
            
        d = snap.to_dict() or {}
        if d.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
            
        ts = d.get("updated_at")
        return SessionOut(
            id=sid,
            messages=[Message(**m) for m in d.get("messages", [])],
            title=d.get("title"),
            updated_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
        )
        
    except Exception as e:
        raise HTTPException(500, f"Error fetching session: {str(e)}")

@router.post("/message")
async def append_message(
    payload: MessageIn, 
    current_email: str = Depends(get_current_email)
):
    try:
        doc_ref = db.collection(COLL).document(payload.session_id)
        snap = await doc_ref.get()  # Make sure to await the get() call
        
        if not snap.exists:
            raise HTTPException(404, "Session not found")
            
        session_data = snap.to_dict() or {}
        if session_data.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")

        # Get existing messages or initialize empty list
        messages = session_data.get("messages", [])
        
        # Append new message
        new_message = {
            "role": payload.role,
            "content": payload.content,
            "type": payload.type,
            "quiz": payload.quiz,
        }
        messages.append(new_message)

        # Generate title from first user message if not exists
        title = session_data.get("title")
        if not title and payload.role == "user":
            clean_content = payload.content.strip()
            title = (clean_content[1:] if clean_content.startswith("@") else clean_content)[:48] or "Session"

        # Prepare update data
        update_data = {
            "messages": messages,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "last_message_preview": payload.content[:160],
        }
        
        # Only update title if we're setting it for the first time
        if not session_data.get("title") and title:
            update_data["title"] = title

        # Perform the update
        await doc_ref.set(update_data, merge=True)

        return {"ok": True}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error saving message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save message. Please try again."
        )

# ---- Provider-agnostic chat (demo/stub) -----------------------------------
# Replace the body of this handler to call OpenAI, Google, etc.
# Replace the /chat endpoint with:
@router.post("/chat")
async def chat_handler(
    data: Dict[str, Any] = Body(...),
    current_email: str = Depends(get_current_email)
):
    session_id: str = data.get("session_id")
    message: str = (data.get("message") or "").strip()
    
    if not session_id:
        raise HTTPException(400, "session_id required")
    if not message:
        raise HTTPException(400, "message required")

    try:
        # Get async document reference
        doc_ref = db.collection(COLL).document(session_id)
        
        # AWAIT the get() operation
        snap = await doc_ref.get()
        
        if not snap.exists:
            raise HTTPException(404, "Session not found")
            
        if snap.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")

        # Get conversation history
        messages = snap.to_dict().get("messages", [])
        
        # Format for Gemini (alternating user/assistant messages)
        history = []
        for msg in messages[-10:]:  # Last 10 messages for context
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})

        # Start chat session with history
        chat = model.start_chat(history=history)
        
        # Get response with retry for rate limits
        @retry.Retry()
        def send_with_retry():
            return chat.send_message(message)
        
        response = send_with_retry()
        reply = response.text

        # Special handling for quiz command
        if message.lower().startswith("@quiz"):
            topic = _topic_from(message)
            quiz = _make_quiz(topic)
            return {
                "reply": "Here's a quick 3-question quiz. Answer and I'll adapt the next one.",
                "type": "quiz",
                "quiz": quiz
            }
            
        return {"reply": reply}

    except Exception as e:
        error_msg = handle_gemini_error(e)
        print(f"Gemini error: {str(e)}")
        return {"reply": error_msg}

# ---- Demo helpers (match your DemoPage) -----------------------------------
def _topic_from(cmd: str) -> str:
    import re
    t = re.sub(r"^@[\w-]+\s*", "", cmd or "").strip()
    return t or "this topic"


def _explain_stub(cmd: str) -> str:
    t = _topic_from(cmd)
    return (
        f"Okay, let's break down {t} in 3 steps:\n\n"
        "1) Intuition — what it is and why it matters.\n"
        "2) The core rule(s) with a tiny example.\n"
        "3) Common mistakes + a practice prompt to try.\n\n"
        f"Want worked examples? Type @examples {t}."
    )


def _hint_stub(cmd: str) -> str:
    t = _topic_from(cmd)
    return (
        f"Hint for {t}: Identify what's given and what's asked. Mark the knowns, then choose a method that connects them. "
        "If stuck, try a simpler version first."
    )


def _summarize_stub(cmd: str) -> str:
    t = _topic_from(cmd)
    return (
        f"Summary of {t}:\n• Key idea in one line.\n• 2–3 rules or patterns to memorize.\n• A minimal example to anchor the concept.\n• One pitfall to avoid."
    )


def _examples_stub(cmd: str) -> str:
    t = _topic_from(cmd)
    return (
        f"Here are two worked examples for {t}:\n\n"
        "Example A → Short, numeric/structured.\n"
        "Example B → Word problem with reasoning.\n\n"
        f"Want a quick check? Type @quiz {t}."
    )

def _make_quiz(topic: str) -> Dict[str, Any]:
    """Generate a relevant quiz about the given topic"""
    topic = topic.lower().strip()
    
    # Science-focused question templates
    science_questions = [
        {
            "template": "What is the primary purpose of {topic}?",
            "options": [
                "Energy production",
                "Information storage", 
                "Structural support",
                "Waste removal"
            ],
            "correct": 0  # Adjust per topic
        },
        {
            "template": "Which of these is essential for {topic}?",
            "options": [
                "Oxygen",
                "Carbon dioxide",
                "Water",
                "Nitrogen"
            ],
            "correct": None  # Will be set based on topic
        },
        {
            "template": "Where does {topic} primarily occur in cells?",
            "options": [
                "Mitochondria",
                "Chloroplasts",
                "Nucleus",
                "Ribosomes"
            ],
            "correct": None
        }
    ]

    # Adjust answers based on topic
    if "photosynthesis" in topic:
        science_questions[1]["correct"] = 1  # CO2 for photosynthesis
        science_questions[2]["correct"] = 1  # Chloroplasts
    elif "respiration" in topic:
        science_questions[1]["correct"] = 0  # Oxygen
        science_questions[2]["correct"] = 0  # Mitochondria
    
    # Build the quiz
    quiz = {
        "topic": topic.capitalize(),
        "items": []
    }
    
    for i, q in enumerate(science_questions[:3]):  # Only take 3 questions
        quiz["items"].append({
            "prompt": q["template"].format(topic=topic),
            "choices": q["options"],
            "answerIdx": q["correct"],
            "feedback": {
                "correct": "Correct! Good understanding of {topic}.".format(topic=topic),
                "wrong": "Review the fundamentals of {topic}.".format(topic=topic)
            }
        })
    
    return quiz
    t = _topic_from(cmd_or_topic)
    return {
      "topic": t,
      "items": [
        {
          "prompt": f"Which statement is true about {t}?",
          "choices": [
            "It always decreases with practice",
            "It's a concept with fixed steps only",
            "It has key rules but also strategy",
            "It cannot be visualized"
          ],
          "answerIdx": 2,
          "feedback": {"correct": "Nice! Understanding rules + strategy matters.", "wrong": "Not quite. Think rules + strategy working together."}
        },
        {
          "prompt": f"Pick the best first step when tackling {t}.",
          "choices": [
            "Guess and check randomly",
            "Identify what's given and what's asked",
            "Memorize the last example",
            "Skip to the final answer"
          ],
          "answerIdx": 1,
          "feedback": {"correct": "Yep—clarify the givens and goal first.", "wrong": "Try to start by clarifying the problem: givens and goal."}
        },
        {
          "prompt": f"You're stuck on {t}. What should you try?",
          "choices": [
            "Close the tab",
            "A simpler analogous problem",
            "Change the topic",
            "Ignore feedback"
          ],
          "answerIdx": 1,
          "feedback": {"correct": "Exactly—reduce complexity and rebuild.", "wrong": "A simpler analogous problem helps you unlock the approach."}
        }
      ]
    }