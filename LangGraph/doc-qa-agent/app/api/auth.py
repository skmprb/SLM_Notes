"""
Auth: extracts user_id from request headers.
Currently uses X-User-ID header — swap this function for JWT decode in production.
"""
from fastapi import Header, HTTPException, status


async def get_current_user(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    """
    Extract user_id from X-User-ID header.
    Production swap: decode JWT, extract sub claim, validate signature.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header is required",
        )
    return x_user_id.strip()
