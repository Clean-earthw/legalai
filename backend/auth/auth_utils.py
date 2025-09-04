# clerk_auth.py
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, jwk
from jose.exceptions import JWTError
import requests
import os
from typing import Dict, Any
from functools import lru_cache
import logging

logger = logging.getLogger("legal-support-api")

class ClerkAuth:
    def __init__(self):
        self.jwks_url = os.getenv("CLERK_JWKS_URL")
        self.issuer = os.getenv("CLERK_ISSUER")
        self.security = HTTPBearer(auto_error=False)
    
    @lru_cache(maxsize=1)
    def get_jwks(self):
        try:
            response = requests.get(self.jwks_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    def get_public_key(self, kid: str):
        jwks = self.get_jwks()
        for key in jwks['keys']:
            if key['kid'] == kid:
                return jwk.construct(key)
        raise HTTPException(status_code=401, detail="Invalid token")
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        try:
            headers = jwt.get_unverified_headers(token)
            kid = headers.get('kid')
            if not kid:
                raise HTTPException(status_code=401, detail="Invalid token format")
                
            public_key = self.get_public_key(kid)
            
            return jwt.decode(
                token, 
                public_key.to_pem().decode('utf-8'), 
                algorithms=['RS256'], 
                audience=self.issuer,
                issuer=self.issuer
            )
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    async def get_current_user(self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        token = credentials.credentials
        try:
            payload = self.decode_token(token)
            
            # Extract user ID from token
            user_id = payload.get('sub')
            if not user_id:
                raise HTTPException(status_code=401, detail="User ID not found in token")
            
            # Add user information to payload
            payload['user_id'] = user_id
            return payload
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected auth error: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")

# Global auth instance
clerk_auth = ClerkAuth()

# Dependency for protected routes
async def get_current_user(auth_data: dict = Depends(clerk_auth.get_current_user)):
    return auth_data