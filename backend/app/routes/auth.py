from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import Client, create_client

router = APIRouter()


# SUPABASE_URL_OLD = 'https://ponbschvojevdnlcxbpb.supabase.co'
# SUPABASE_KEY_OLD = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvbmJzY2h2b2pldmRubGN4YnBiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjc2MTQ0OTUsImV4cCI6MjA0MzE5MDQ5NX0.2vNYFiTycDCKXSTJ-SXJ0WePIJcgB2eL4uhFR09KFfM'


# Initialize Supabase client
SUPABASE_URL = "https://ponbschvojevdnlcxbpb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBvbmJzY2h2b2pldmRubGN4YnBiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjc2MTQ0OTUsImV4cCI6MjA0MzE5MDQ5NX0.2vNYFiTycDCKXSTJ-SXJ0WePIJcgB2eL4uhFR09KFfM"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class UserAuth(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def signup(user: UserAuth):
    try:
        response = supabase.auth.sign_up(
            {"email": user.email, "password": user.password}
        )
        return {"message": "User created successfully", "user": response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(user: UserAuth):
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": user.email, "password": user.password}
        )
        return {"access_token": response.session.access_token, "user": response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
