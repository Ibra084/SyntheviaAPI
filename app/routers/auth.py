from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from firebase_admin import auth as firebase_auth
from app.models.schemas import Token, UserCreate
from app.services.auth_service import register_user, login_user, social_login_from_claims

router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.post("/register", response_model=Token)
def register(payload: UserCreate):
    token = register_user(payload.name, payload.email, payload.password)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    token = login_user(form.username, form.password)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/google", response_model=Token)
def google_auth(token: str = Body(..., embed=True)):
    try:
        decoded = firebase_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token")
    email = decoded.get("email")
    name = decoded.get("name")
    access = social_login_from_claims(email, name)
    return {"access_token": access, "token_type": "bearer"}

@router.post("/github", response_model=Token)
def github_auth(token: str = Body(..., embed=True)):
    try:
        decoded = firebase_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token")
    email = decoded.get("email")
    name = decoded.get("name")
    access = social_login_from_claims(email, name)
    return {"access_token": access, "token_type": "bearer"}
