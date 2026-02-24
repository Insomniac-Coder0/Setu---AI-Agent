from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from typing import Dict
from urllib.parse import urlencode
from app.api import deps
from app.core.config import settings
from app.core.database import save_user_credentials
import httpx
import secrets
import base64
import os

def _get_frontend_url() -> str:
    """Get the frontend URL for OAuth redirects."""
    return os.getenv("FRONTEND_URL", "http://localhost:5173")

router = APIRouter()

# Store state → user_id mapping (in production, use Redis)
oauth_states = {}

@router.get("/google/login")
async def login_google(current_user: dict = Depends(deps.get_current_user)):
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Google credentials not configured")
    
    # Generate state to prevent CSRF
    state = secrets.token_urlsafe(32)
    user_id = current_user.get('sub') or current_user.get('id', 'unknown')
    oauth_states[state] = user_id
    
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets",
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    return {"auth_url": f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}", "state": state}

@router.get("/google/callback")
async def callback_google(code: str, state: str):
    # Verify state
    user_id = oauth_states.pop(state, None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    # Exchange code for access token
    token_url = "https://oauth2.googleapis.com/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data={
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        })
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {response.text}")
    
    token_data = response.json()
    
    # Save credentials
    await save_user_credentials(
        user_id=user_id,
        service_name="Google",
        service_type="google",
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        expires_in=token_data.get("expires_in", 3600),
        scopes=["gmail", "calendar", "drive"]
    )
    
    # Redirect back to frontend
    return RedirectResponse(url=f"{_get_frontend_url()}?success=google_connected")

@router.get("/notion/login")
async def login_notion(current_user: dict = Depends(deps.get_current_user)):
    if not settings.NOTION_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Notion credentials not configured")
    
    state = secrets.token_urlsafe(32)
    user_id = current_user.get('sub') or current_user.get('id', 'unknown')
    oauth_states[state] = user_id
    
    params = {
        "client_id": settings.NOTION_CLIENT_ID,
        "redirect_uri": settings.NOTION_REDIRECT_URI,
        "response_type": "code",
        "owner": "user",
        "state": state
    }
    return {"auth_url": f"https://api.notion.com/v1/oauth/authorize?{urlencode(params)}", "state": state}

@router.get("/notion/callback")
async def callback_notion(code: str, state: str):
    user_id = oauth_states.pop(state, None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    # Exchange code for token
    token_url = "https://api.notion.com/v1/oauth/token"
    
    async with httpx.AsyncClient() as client:
        auth = base64.b64encode(f"{settings.NOTION_CLIENT_ID}:{settings.NOTION_CLIENT_SECRET}".encode()).decode()
        response = await client.post(
            token_url,
            headers={"Authorization": f"Basic {auth}"},
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.NOTION_REDIRECT_URI
            }
        )
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Token exchange failed")
    
    token_data = response.json()
    
    await save_user_credentials(
        user_id=user_id,
        service_name="Notion",
        service_type="notion",
        access_token=token_data["access_token"],
        expires_in=31536000,  # Notion tokens don't expire
        scopes=["read", "write"]
    )
    
    return RedirectResponse(url=f"{_get_frontend_url()}?success=notion_connected")

@router.get("/github/login")
async def login_github(current_user: dict = Depends(deps.get_current_user)):
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GitHub not configured")
    
    state = secrets.token_urlsafe(32)
    user_id = current_user.get('sub') or current_user.get('id', 'unknown')
    oauth_states[state] = user_id
    
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "scope": "repo user",
        "state": state
    }
    return {"auth_url": f"https://github.com/login/oauth/authorize?{urlencode(params)}", "state": state}

@router.get("/github/callback")
async def callback_github(code: str, state: str):
    user_id = oauth_states.pop(state, None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code
            }
        )
    
    token_data = response.json()
    
    await save_user_credentials(
        user_id=user_id,
        service_name="GitHub",
        service_type="github",
        access_token=token_data["access_token"],
        scopes=["repo", "user"]
    )
    
    return RedirectResponse(url=f"{_get_frontend_url()}?success=github_connected")

@router.get("/slack/login")
async def login_slack(current_user: dict = Depends(deps.get_current_user)):
    if not settings.SLACK_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Slack credentials not configured")
    
    state = secrets.token_urlsafe(32)
    user_id = current_user.get('sub') or current_user.get('id', 'unknown')
    oauth_states[state] = user_id
    
    params = {
        "client_id": settings.SLACK_CLIENT_ID,
        "scope": "chat:write,commands", 
        "user_scope": "",
        "state": state
    }
    return {"auth_url": f"https://slack.com/oauth/v2/authorize?{urlencode(params)}", "state": state}

@router.get("/slack/callback")
async def callback_slack(code: str, state: str):
    user_id = oauth_states.pop(state, None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/oauth.v2.access",
            data={
                "client_id": settings.SLACK_CLIENT_ID,
                "client_secret": settings.SLACK_CLIENT_SECRET,
                "code": code
            }
        )
    
    token_data = response.json()
    
    if not token_data.get("ok"):
        raise HTTPException(status_code=400, detail="Slack token exchange failed")
    
    await save_user_credentials(
        user_id=user_id,
        service_name="Slack",
        service_type="slack",
        access_token=token_data["authed_user"]["access_token"],
        scopes=["chat:write", "commands"]
    )
    
    return RedirectResponse(url=f"{_get_frontend_url()}?success=slack_connected")
