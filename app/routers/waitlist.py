# app/routers/waitlist.py
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.db.firestore import db
from app.services.email_service import send_email
from email.message import EmailMessage
from app.core.config import settings
from google.cloud import firestore as gfs
from fastapi import HTTPException

router = APIRouter(prefix="/api/waitlist", tags=["Waitlist"])


class WaitlistEntry(BaseModel):
    name: str
    email: EmailStr
    background: Optional[str] = None
    motivation: Optional[str] = None

class WaitlistResponse(BaseModel):
    message: str
    position: int
    email: EmailStr

@router.post("", response_model=WaitlistResponse)
async def join_waitlist(entry: WaitlistEntry, bg: BackgroundTasks):
    # prevent duplicates
    existing = list(db.collection("waitlist").where("email","==",entry.email).limit(1).stream())
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered for waitlist")
        

    # position = count + 1
    all_docs = list(db.collection("waitlist").stream())
    position = len(all_docs) + 1

    data = {
        "name": entry.name,
        "email": entry.email,
        "background": entry.background,
        "motivation": entry.motivation,
        "joined_at": datetime.utcnow(),
        "position": position,
        "notified": False,
        "status": "waiting",
    }
    db.collection("waitlist").document(entry.email).set(data)

    # confirmation email (queued)
    user_msg = EmailMessage()
    user_msg["From"] = settings.GMAIL_USER
    user_msg["To"] = entry.email
    user_msg["Subject"] = "🎯 You're on the AI Mastery Waitlist!"
    user_msg.set_content(f"Hi {entry.name}, you're #{position} on the waitlist.")
    bg.add_task(send_email, user_msg)

    # admin notify (queued)
    admin_msg = EmailMessage()
    admin_msg["From"] = settings.GMAIL_USER
    admin_msg["To"] = "ibrahim.rahman@synthevia.com"
    admin_msg["Subject"] = f"🔥 New Waitlist Signup #{position} - {entry.name}"
    admin_msg.set_content(f"{entry.name} <{entry.email}>\nBackground: {entry.background}\nMotivation: {entry.motivation}")
    bg.add_task(send_email, admin_msg)


    return WaitlistResponse(
        message=f"Successfully joined the waitlist! You're #{position}.",
        position=position,
        email=entry.email,
    )

@router.get("")
async def list_waitlist(limit: int = 50, offset: int = 0, status_filter: Optional[str] = None):
    q = db.collection("waitlist").order_by("joined_at", direction=gfs.Query.DESCENDING)
    if status_filter:
        q = q.where("status","==",status_filter)
    docs = list(q.offset(offset).limit(limit).stream())
    out = []
    for d in docs:
        itm = d.to_dict()
        if "joined_at" in itm and hasattr(itm["joined_at"], "isoformat"):
            itm["joined_at"] = itm["joined_at"].isoformat()
        out.append(itm)
    return {"entries": out, "count": len(out), "offset": offset, "limit": limit}
