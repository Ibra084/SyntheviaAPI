# app/routers/community.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from app.db.firestore import db
from google.cloud import firestore

router = APIRouter(prefix="/api/community", tags=["Community"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

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

@router.get("/leaderboard")
def get_leaderboard(
    filter_by: str = Query("points", regex="^(points|modules|streak)$"),
    limit: int = Query(20, ge=1, le=100),
    _: str = Depends(get_current_email)  # auth gate
):
    users_ref = db.collection("user_progress")

    # ⚠️ Firestore cannot order by array length; prefer a stored counter field.
    if filter_by == "points":
        query = users_ref.order_by("total_points", direction=firestore.Query.DESCENDING).limit(limit)
        docs = list(query.stream())
    elif filter_by == "streak":
        query = users_ref.order_by("streak", direction=firestore.Query.DESCENDING).limit(limit)
        docs = list(query.stream())
    else:  # modules
        # Try to use a count field if you have it; otherwise fallback to Python sort.
        try:
            query = users_ref.order_by("completed_modules_count", direction=firestore.Query.DESCENDING).limit(limit)
            docs = list(query.stream())
        except Exception:
            # Fallback: fetch top by points (cheap) then sort by completed_modules length
            docs = list(users_ref.order_by("total_points", direction=firestore.Query.DESCENDING).limit(200).stream())
            docs.sort(key=lambda d: len(d.to_dict().get("completed_modules", [])), reverse=True)
            docs = docs[:limit]

    leaderboard = []
    for d in docs:
        u = d.to_dict() or {}
        email = u.get("user_email", "")

        # Join with users collection
        profile_doc = db.collection("users").document(email).get()
        if not profile_doc.exists:
            continue  # skip deleted users

        profile = profile_doc.to_dict() or {}
        if not profile.get("is_active", True):
            continue  # skip deactivated users

        # optional: also allow an explicit flag
        if profile.get("include_in_leaderboard") is False:
            continue

        leaderboard.append({
            "email": email,
            "name": profile.get("name", "User"),
            "total_points": u.get("total_points", 0),
            "completed_modules": len(u.get("completed_modules", [])),
            "streak": u.get("streak", 0),
            "current_module": u.get("current_module", 1),
            "last_active": u.get("last_active_date"),
        })

    return leaderboard
