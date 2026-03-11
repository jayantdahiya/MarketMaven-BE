"""
AuthService: Supabase signup, login, logout.
"""
from supabase import create_client, Client


class AuthService:
    def __init__(self, supabase_url: str, supabase_key: str):
        self._client: Client = create_client(supabase_url, supabase_key)

    def signup(self, email: str, password: str) -> dict:
        try:
            result = self._client.auth.sign_up({"email": email, "password": password})
            user = result.model_dump() if hasattr(result, "model_dump") else result
            return {
                "user_id": user.get("user", {}).get("id") if isinstance(user.get("user"), dict) else str(getattr(user.get("user"), "id", "")),
                "access_token": user.get("session", {}).get("access_token") if isinstance(user.get("session"), dict) else getattr(user.get("session"), "access_token", None),
                "message": "Signup successful",
            }
        except Exception as e:
            return {"user_id": None, "access_token": None, "message": str(e)}

    def login(self, email: str, password: str) -> dict:
        try:
            result = self._client.auth.sign_in_with_password({"email": email, "password": password})
            session = result.session
            return {
                "user_id": result.user.id if result.user else None,
                "access_token": session.access_token if session else None,
                "message": "Login successful",
            }
        except Exception as e:
            return {"user_id": None, "access_token": None, "message": str(e)}

    def logout(self) -> dict:
        try:
            self._client.auth.sign_out()
            return {"message": "Signed out"}
        except Exception as e:
            return {"message": str(e)}
