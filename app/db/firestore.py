import os
from google.cloud import firestore
import firebase_admin
from firebase_admin import initialize_app
from app.core.config import settings

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    settings.GOOGLE_APPLICATION_CREDENTIALS
)

if not firebase_admin._apps:
    initialize_app()

db = firestore.Client()

def ensure_weak_areas_index():
    """Ensure the weak areas collection has the right indexes."""
    # Firestore automatically creates single-field indexes
    # For composite indexes (e.g., severity + last_encountered), you can set them up in the console
    pass

# Run at startup
ensure_weak_areas_index()
