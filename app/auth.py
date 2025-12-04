import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth, OAuthError

router = APIRouter(prefix="/auth", tags=["auth"])

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def _redirect_uri(request: Request) -> str:
    # Prefer explicit env; fallback to URL based on current request
    env_uri = os.environ.get("GOOGLE_REDIRECT_URI")
    if env_uri:
        return env_uri
    # Build absolute callback URL
    url = request.url_for("auth_callback")
    # Ensure https if behind proxy can be handled later; for local dev http is fine
    return str(url)


@router.get("/login")
async def auth_login(request: Request):
    if not oauth.google.client_id or not oauth.google.client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
    redirect_uri = _redirect_uri(request)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(request: Request):
    try:
        print(f"DEBUG: Session keys: {request.session.keys()}")
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {e.error}")

    # Prefer ID Token claims
    id_token = token.get("userinfo") or token.get("id_token")
    userinfo = None
    if isinstance(id_token, dict):
        userinfo = id_token
    else:
        # Fallback: fetch userinfo endpoint
        try:
            userinfo = await oauth.google.parse_id_token(request, token)
        except Exception:
            resp = await oauth.google.get("userinfo", token=token)
            userinfo = resp.json()

    if not userinfo:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")

    # Store a minimal user profile in session
    request.session["user"] = {
        "sub": userinfo.get("sub"),
        "email": userinfo.get("email"),
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
    }
    return RedirectResponse(url="/auth/me")


@router.get("/me")
async def me(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse(user)


@router.post("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return JSONResponse({"ok": True})
