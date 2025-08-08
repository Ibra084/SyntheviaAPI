from fastapi import HTTPException, status
from datetime import datetime, timezone
from app.core.security import hash_password, verify_password, create_access_token
from app.repositories.users_repo import get_user, create_user, update_last_login

def register_user(name: str, email: str, password: str) -> str:
    if get_user(email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = {
        "name": name,
        "email": email,
        "hashed_password": hash_password(password),
        "created_at": datetime.now(timezone.utc),
        "is_active": True,
    }
    create_user(user)
    return create_access_token(sub=email)

def login_user(email: str, password: str) -> str:
    user = get_user(email)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    update_last_login(email)
    return create_access_token(sub=email)

def social_login_from_claims(email: str, name: str | None) -> str:
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token (no email)")
    user = get_user(email)
    if not user:
        create_user({
            "name": name or email.split("@")[0],
            "email": email,
            "hashed_password": hash_password("*" * 32),
            "created_at": datetime.now(timezone.utc),
            "is_active": True,
        })
    else:
        update_last_login(email)
    return create_access_token(sub=email)
