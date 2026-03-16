"""
Auth routes: signup, login, logout.
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas.auth import AuthResponse, LoginRequest, SignUpRequest

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/signup', response_model=AuthResponse)
async def signup(req: SignUpRequest, request: Request) -> AuthResponse:
    service = getattr(request.app.state, 'auth_service', None)
    if service is None:
        raise HTTPException(
            status_code=503, detail='Auth service unavailable (missing Supabase)'
        )
    out = service.signup(req.email, req.password)
    if (
        out.get('access_token') is None
        and 'error' in str(out.get('message', '')).lower()
    ):
        raise HTTPException(status_code=400, detail=out.get('message', 'Signup failed'))
    return AuthResponse(**out)


@router.post('/login', response_model=AuthResponse)
async def login(req: LoginRequest, request: Request) -> AuthResponse:
    service = getattr(request.app.state, 'auth_service', None)
    if service is None:
        raise HTTPException(
            status_code=503, detail='Auth service unavailable (missing Supabase)'
        )
    out = service.login(req.email, req.password)
    if out.get('access_token') is None:
        raise HTTPException(status_code=401, detail='Invalid credentials')
    return AuthResponse(**out)


@router.get('/logout')
async def logout(request: Request) -> dict:
    service = getattr(request.app.state, 'auth_service', None)
    if service is None:
        return {'message': 'Auth service unavailable'}
    return service.logout()
