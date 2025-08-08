import os, json, base64, pathlib
from functools import lru_cache

from google.cloud import firestore
from google.oauth2 import service_account

import firebase_admin
from firebase_admin import initialize_app
from app.core.config import settings

# Don't override GOOGLE_APPLICATION_CREDENTIALS here.
# In Render, set it in the dashboard to: /etc/secrets/firebase-key.json

if not firebase_admin._apps:
    initialize_app()

@lru_cache(maxsize=1)
def get_db():
    """Create Firestore client once, safely in any env."""
    # 1) Prefer base64 secret if present (optional but nice to have)
    key_b64 = os.getenv("FIREBASE_KEY_B64")
    if key_b64:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(base64.b64decode(key_b64))
        )
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or creds.project_id
        return firestore.Client(project=project, credentials=creds)

    # 2) Otherwise use a file path from env or settings
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or settings.GOOGLE_APPLICATION_CREDENTIALS
    if path and pathlib.Path(path).exists():
        creds = service_account.Credentials.from_service_account_file(path)
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or creds.project_id
        return firestore.Client(project=project, credentials=creds)

    # 3) Fall back to ADC (works locally if you’ve run `gcloud auth application-default login`)
    return firestore.Client()

# Use get_db() wherever you need Firestore
db = get_db()
