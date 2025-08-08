from fastapi import APIRouter
from app.db.firestore import db

router = APIRouter(prefix="/api/modules", tags=["Modules"])

@router.get("")
def get_modules():
    docs = db.collection("modules").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]
