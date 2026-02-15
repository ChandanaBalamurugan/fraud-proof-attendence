import os
import firebase_admin
from firebase_admin import credentials, firestore


def init_firebase(service_account_path=None):
    """Initialize Firebase app and return Firestore client.
    If service_account_path is None, will try to initialize using
    the GOOGLE_APPLICATION_CREDENTIALS env var or an existing app.
    Returns None if initialization fails or service account not found.
    """
    try:
        # Avoid double-initialization
        if not firebase_admin._apps:
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
            else:
                # Try default application credentials
                try:
                    firebase_admin.initialize_app()
                except Exception:
                    return None
        return firestore.client()
    except Exception:
        return None


def get_firestore_client():
    """Convenience wrapper: looks for `backend/firebase-service-account.json` first."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base, 'firebase-service-account.json')
    candidate = os.path.normpath(candidate)
    if os.path.exists(candidate):
        return init_firebase(candidate)
    # Try environment-based credentials
    return init_firebase()
