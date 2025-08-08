from datetime import datetime, timezone
from app.db.firestore import db

def get_user(email: str) -> dict | None:
    doc = db.collection("users").document(email).get()
    return doc.to_dict() if doc.exists else None

def create_user(user: dict) -> None:
    db.collection("users").document(user["email"]).set(user)

def update_last_login(email: str) -> None:
    db.collection("users").document(email).update({"last_login": datetime.now(timezone.utc)})
