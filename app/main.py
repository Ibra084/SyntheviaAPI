from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import auth, misc, modules, progress, quiz, waitlist, users, community, atlas


if not settings.SECRET_KEY:
    raise RuntimeError("SECRET_KEY is required")

app = FastAPI(title="AI Learning Portal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.synthevia.academy",
        "https://synthevia.academy",
        "https://atlas.synthevia.academy",   # added Atlas frontend
        "https://api.synthevia.academy",     # API origin (if ever called from browser)
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,           # needed if you send cookies/Authorization
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],              # or enumerate what you actually use
    expose_headers=["*"],             # optional
)

app.include_router(auth.router)
app.include_router(misc.router)
app.include_router(modules.router)
app.include_router(progress.router)
app.include_router(quiz.router)
app.include_router(waitlist.router)
app.include_router(users.router)
app.include_router(community.router)
app.include_router(atlas.router)

@app.get("/")
def root():
    return {"message": "AI Learning Portal API"}
