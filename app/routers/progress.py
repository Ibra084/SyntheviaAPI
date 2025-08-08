from fastapi import APIRouter

router = APIRouter(prefix="/api/progress", tags=["Progress"])

@router.get("")
def get_progress_placeholder():
    return {"message": "progress endpoint placeholder"}
