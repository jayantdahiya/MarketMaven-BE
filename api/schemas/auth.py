"""
Auth request/response schemas.
"""
from pydantic import BaseModel, EmailStr, Field


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class AuthResponse(BaseModel):
    user_id: str | None = None
    access_token: str | None = None
    message: str
