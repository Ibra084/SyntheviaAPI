from fastapi import APIRouter

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])

@router.get("")
def get_quiz_placeholder():
    return {"message": "quiz endpoint placeholder"}
