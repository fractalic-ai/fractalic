"""
Enhanced OAuth Token Validation and Expiration Detection for MCP Services
Based on MCP SDK documentation and OAuth 2.1 specifications
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class TokenValidationResult:
    """Result of token validation with detailed status"""
    is_valid: bool
    is_expired: bool
    needs_refresh: bool
    error_type: Optional[str] = None
    error_description: Optional[str] = None
    expires_at: Optional[datetime] = None
    time_until_expiry: Optional[int] = None

class EnhancedTokenManager:
    """Enhanced token management with proper expiration detection and validation"""
    
    def __init__(self, token_file_path: str):
        self.token_file_path = token_file_path
    
    def _calculate_expires_at(self, expires_in: int) -> datetime:
        """Calculate absolute expiration time from expires_in seconds"""
        return datetime.now() + timedelta(seconds=expires_in)
    
    def save_token_with_expiration(self, service_name: str, token_data: Dict[str, Any]) -> None:
        """Save token with calculated expiration timestamp"""
        try:
            # Load existing tokens
            data = {}
            try:
                with open(self.token_file_path, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                pass
            
            # Calculate expires_at if not present
            if 'expires_in' in token_data and 'expires_at' not in token_data:
                expires_at = self._calculate_expires_at(token_data['expires_in'])
                token_data['expires_at'] = expires_at.isoformat()
                token_data['created_at'] = datetime.now().isoformat()
            
            # Save updated token data
            data[service_name] = token_data
            
            with open(self.token_file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Enhanced token saved for {service_name} with expiration tracking")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced token for {service_name}: {e}")
    
    def validate_token(self, service_name: str) -> TokenValidationResult:
        """Comprehensive token validation per MCP/OAuth 2.1 specifications"""
        try:
            # Load token data
            with open(self.token_file_path, 'r') as f:
                data = json.load(f)
            
            if service_name not in data:
                return TokenValidationResult(
                    is_valid=False,
                    is_expired=False,
                    needs_refresh=False,
                    error_type="missing_token",
                    error_description="No token found for service"
                )
            
            token_data = data[service_name]
            access_token = token_data.get('access_token')
            
            if not access_token:
                return TokenValidationResult(
                    is_valid=False,
                    is_expired=False,
                    needs_refresh=False,
                    error_type="invalid_token",
                    error_description="Missing access_token"
                )
            
            # Check expiration using expires_at if available
            expires_at_str = token_data.get('expires_at')
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                now = datetime.now()
                
                if now >= expires_at:
                    return TokenValidationResult(
                        is_valid=False,
                        is_expired=True,
                        needs_refresh=bool(token_data.get('refresh_token')),
                        error_type="expired_token",
                        error_description="Token has expired",
                        expires_at=expires_at,
                        time_until_expiry=0
                    )
                
                # Token is valid but check if it expires soon (within 5 minutes)
                time_until_expiry = int((expires_at - now).total_seconds())
                needs_refresh = time_until_expiry < 300  # Refresh if expires in < 5 min
                
                return TokenValidationResult(
                    is_valid=True,
                    is_expired=False,
                    needs_refresh=needs_refresh,
                    expires_at=expires_at,
                    time_until_expiry=time_until_expiry
                )
            else:
                # Fallback: check using expires_in + created_at or issue warning
                logger.warning(f"No expires_at timestamp for {service_name}, token validation incomplete")
                return TokenValidationResult(
                    is_valid=True,  # Assume valid, will be caught by API call
                    is_expired=False,
                    needs_refresh=False,
                    error_type="validation_incomplete",
                    error_description="Cannot determine expiration without expires_at timestamp"
                )
                
        except Exception as e:
            logger.error(f"Token validation failed for {service_name}: {e}")
            return TokenValidationResult(
                is_valid=False,
                is_expired=False,
                needs_refresh=False,
                error_type="validation_error",
                error_description=f"Validation error: {str(e)}"
            )
    
    async def validate_token_with_api_call(self, service_name: str, api_url: str) -> TokenValidationResult:
        """Validate token by making actual API call (per MCP spec - 401 detection)"""
        # First do local validation
        local_validation = self.validate_token(service_name)
        if not local_validation.is_valid:
            return local_validation
        
        try:
            # Load token for API call
            with open(self.token_file_path, 'r') as f:
                data = json.load(f)
            
            token_data = data[service_name]
            access_token = token_data['access_token']
            
            # Make test API call with Bearer token (per MCP spec)
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(api_url, headers=headers, timeout=5.0)
                
                if response.status_code == 401:
                    # Per MCP spec: "Invalid or expired tokens MUST receive a HTTP 401 response"
                    error_details = self._parse_401_error(response)
                    return TokenValidationResult(
                        is_valid=False,
                        is_expired=True,  # 401 typically means expired or invalid
                        needs_refresh=bool(token_data.get('refresh_token')),
                        error_type="api_unauthorized",
                        error_description=f"API returned 401: {error_details}"
                    )
                elif response.status_code == 403:
                    # Per MCP spec: 403 = "Invalid scopes or insufficient permissions"
                    return TokenValidationResult(
                        is_valid=False,
                        is_expired=False,
                        needs_refresh=False,
                        error_type="insufficient_permissions",
                        error_description="API returned 403: Insufficient permissions or invalid scopes"
                    )
                elif 200 <= response.status_code < 300:
                    # Token is valid
                    return TokenValidationResult(
                        is_valid=True,
                        is_expired=False,
                        needs_refresh=local_validation.needs_refresh,
                        expires_at=local_validation.expires_at,
                        time_until_expiry=local_validation.time_until_expiry
                    )
                else:
                    # Other error
                    return TokenValidationResult(
                        is_valid=False,
                        is_expired=False,
                        needs_refresh=False,
                        error_type="api_error",
                        error_description=f"API returned {response.status_code}: {response.text[:200]}"
                    )
                    
        except Exception as e:
            logger.error(f"API validation failed for {service_name}: {e}")
            return TokenValidationResult(
                is_valid=False,
                is_expired=False,
                needs_refresh=False,
                error_type="api_validation_error",
                error_description=f"API validation error: {str(e)}"
            )
    
    def _parse_401_error(self, response: httpx.Response) -> str:
        """Parse 401 error response to extract specific error details"""
        try:
            # Check for WWW-Authenticate header (per OAuth 2.1)
            www_auth = response.headers.get('WWW-Authenticate', '')
            if www_auth:
                return f"WWW-Authenticate: {www_auth}"
            
            # Try to parse JSON error response
            try:
                error_data = response.json()
                error = error_data.get('error', 'unauthorized')
                error_description = error_data.get('error_description', '')
                return f"{error}: {error_description}" if error_description else error
            except:
                pass
            
            # Fallback to response text
            return response.text[:200] if response.text else "Unauthorized"
            
        except Exception:
            return "Unauthorized (unable to parse error details)"
    
    async def refresh_token_if_needed(self, service_name: str, token_endpoint: str) -> bool:
        """Attempt to refresh token using refresh_token (per OAuth 2.1)"""
        try:
            validation = self.validate_token(service_name)
            if not validation.needs_refresh and validation.is_valid:
                return True  # No refresh needed
            
            # Load current token data
            with open(self.token_file_path, 'r') as f:
                data = json.load(f)
            
            token_data = data[service_name]
            refresh_token = token_data.get('refresh_token')
            
            if not refresh_token:
                logger.warning(f"No refresh token available for {service_name}")
                return False
            
            # Make refresh token request
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_endpoint,
                    data=refresh_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    new_token_data = response.json()
                    # Update token with new expiration
                    self.save_token_with_expiration(service_name, new_token_data)
                    logger.info(f"Successfully refreshed token for {service_name}")
                    return True
                else:
                    logger.error(f"Token refresh failed for {service_name}: {response.status_code} {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Token refresh error for {service_name}: {e}")
            return False

# Usage Example for Replicate service validation
async def validate_replicate_token():
    """Example of how to validate Replicate token properly"""
    token_manager = EnhancedTokenManager("oauth_tokens.json")
    
    # Step 1: Local validation (check expiration timestamp)
    validation = token_manager.validate_token("replicate")
    print(f"Local validation: {validation}")
    
    if validation.is_expired:
        print("Token is expired locally - need to refresh or re-authenticate")
        return
    
    # Step 2: API validation (actual HTTP call to detect 401)
    api_validation = await token_manager.validate_token_with_api_call(
        "replicate", 
        "https://mcp.replicate.com/sse"
    )
    print(f"API validation: {api_validation}")
    
    if api_validation.error_type == "api_unauthorized":
        print("API returned 401 - token is invalid/expired")
        # Attempt refresh or trigger re-authentication
    elif api_validation.is_valid:
        print("Token is valid and working")
    else:
        print(f"Validation failed: {api_validation.error_description}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(validate_replicate_token())
