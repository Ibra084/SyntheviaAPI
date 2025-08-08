# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from app.repositories.users_repo import get_user
from app.models.schemas import UserPublic

router = APIRouter(prefix="/api/users", tags=["Users"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    try:
        payload = decode_token(token)  # {"sub": email, "exp": ...}
        email = payload.get("sub")
        if not email:
            raise ValueError("No subject in token")
        data = get_user(email)
        if not data:
            raise ValueError("User not found")
        return UserPublic(
            name=data["name"],
            email=data["email"],
            created_at=data["created_at"],
            last_login=data.get("last_login"),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me", response_model=UserPublic)
def read_me(current_user: UserPublic = Depends(get_current_user)):
    return current_user
