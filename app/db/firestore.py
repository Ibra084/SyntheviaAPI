import os
from google.cloud import firestore
import firebase_admin
from firebase_admin import initialize_app
from app.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

if not firebase_admin._apps:
    initialize_app()

db = firestore.Client()
