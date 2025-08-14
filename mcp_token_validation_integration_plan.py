"""
Integration Plan: Enhanced OAuth Token Validation for MCP Manager SDK v2

Based on MCP SDK documentation research, here's how to properly implement
token validation and expiration detection in the existing system.
"""

# STEP 1: Enhance Token Storage Format
# Current format in oauth_tokens.json:
CURRENT_FORMAT = {
    "replicate": {
        "access_token": "replicate-dimusdim:...",
        "token_type": "Bearer", 
        "expires_in": 3600,  # ❌ Only relative time, no absolute expiration
        "refresh_token": "replicate-dimusdim:...",
        "scope": "read write"
    }
}

# Enhanced format needed:
ENHANCED_FORMAT = {
    "replicate": {
        "access_token": "replicate-dimusdim:...",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires_at": "2025-08-14T06:31:20.123456",  # ✅ Absolute expiration time
        "created_at": "2025-08-14T05:31:20.123456",  # ✅ Token creation time
        "refresh_token": "replicate-dimusdim:...",
        "scope": "read write",
        "last_validated": "2025-08-14T05:31:25.123456"  # ✅ Last successful validation
    }
}

# STEP 2: Integration Points in fractalic_mcp_manager_sdk_v2.py

INTEGRATION_POINTS = [
    {
        "location": "FileTokenStorage.set_tokens()",
        "current_line": "~128",
        "change": "Add expires_at calculation when saving tokens",
        "code_change": """
        # BEFORE:
        data[self.service_name] = {
            'access_token': tokens.access_token,
            'token_type': tokens.token_type,
            'expires_in': tokens.expires_in,
            'refresh_token': tokens.refresh_token,
            'scope': tokens.scope
        }
        
        # AFTER:
        from datetime import datetime, timedelta
        expires_at = datetime.now() + timedelta(seconds=tokens.expires_in or 3600)
        
        data[self.service_name] = {
            'access_token': tokens.access_token,
            'token_type': tokens.token_type,
            'expires_in': tokens.expires_in,
            'expires_at': expires_at.isoformat(),
            'created_at': datetime.now().isoformat(),
            'refresh_token': tokens.refresh_token,
            'scope': tokens.scope
        }
        """
    },
    {
        "location": "_get_tools_for_service_impl() SSE connection",
        "current_line": "~540",
        "change": "Add token validation before connection attempt",
        "code_change": """
        # BEFORE: Direct token usage
        if tokens:
            headers['Authorization'] = f'Bearer {tokens.access_token}'
        
        # AFTER: Validate token first
        if tokens:
            validation_result = await self._validate_token_with_expiration(service_name, tokens)
            if validation_result.is_expired:
                logger.warning(f"Token expired for {service_name}, attempting refresh...")
                refreshed = await self._attempt_token_refresh(service_name)
                if not refreshed:
                    logger.error(f"Token refresh failed for {service_name}, triggering OAuth flow")
                    # Clear expired token and trigger OAuth
                    await token_storage.set_tokens(None)  # Clear expired token
                    # Continue with OAuth flow...
                else:
                    # Reload refreshed token
                    tokens = await token_storage.get_tokens()
            
            if tokens:
                headers['Authorization'] = f'Bearer {tokens.access_token}'
        """
    },
    {
        "location": "Error handling in connection attempts", 
        "current_line": "Multiple locations",
        "change": "Detect and categorize 401 errors specifically",
        "code_change": """
        # Add new method to detect 401 responses:
        async def _handle_connection_error(self, service_name: str, error: Exception) -> str:
            '''Categorize connection errors per MCP spec'''
            error_str = str(error).lower()
            
            # Check for HTTP 401 Unauthorized (per MCP spec)
            if '401' in error_str or 'unauthorized' in error_str:
                logger.warning(f"Received 401 Unauthorized for {service_name} - token invalid/expired")
                # Attempt to clear expired token
                token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                await token_storage.set_tokens(None)
                return "oauth_expired"
            
            # Check for HTTP 403 Forbidden (insufficient permissions)
            elif '403' in error_str or 'forbidden' in error_str:
                return "insufficient_permissions"
            
            # Other connection errors
            elif 'timeout' in error_str or 'connection' in error_str:
                return "connection_error"
            
            else:
                return "unknown_error"
        """
    }
]

# STEP 3: Add Token Validation Methods

TOKEN_VALIDATION_METHODS = """
# Add these methods to MCPSupervisorV2 class:

async def _validate_token_with_expiration(self, service_name: str, tokens) -> 'TokenValidationResult':
    '''Validate token expiration using expires_at timestamp'''
    try:
        # Load token data to check expires_at
        token_storage = FileTokenStorage("oauth_tokens.json", service_name)
        with open("oauth_tokens.json", 'r') as f:
            data = json.load(f)
        
        if service_name not in data:
            return TokenValidationResult(is_valid=False, is_expired=False, needs_refresh=False)
        
        token_data = data[service_name]
        expires_at_str = token_data.get('expires_at')
        
        if expires_at_str:
            from datetime import datetime
            expires_at = datetime.fromisoformat(expires_at_str)
            now = datetime.now()
            
            if now >= expires_at:
                logger.info(f"Token expired for {service_name} at {expires_at}")
                return TokenValidationResult(
                    is_valid=False, 
                    is_expired=True, 
                    needs_refresh=bool(token_data.get('refresh_token'))
                )
            
            # Check if expires soon (within 5 minutes)
            time_until_expiry = (expires_at - now).total_seconds()
            needs_refresh = time_until_expiry < 300
            
            return TokenValidationResult(
                is_valid=True,
                is_expired=False, 
                needs_refresh=needs_refresh
            )
        else:
            # No expires_at timestamp - assume valid but log warning
            logger.warning(f"No expiration timestamp for {service_name}, cannot validate expiration")
            return TokenValidationResult(is_valid=True, is_expired=False, needs_refresh=False)
            
    except Exception as e:
        logger.error(f"Token validation failed for {service_name}: {e}")
        return TokenValidationResult(is_valid=False, is_expired=False, needs_refresh=False)

async def _attempt_token_refresh(self, service_name: str) -> bool:
    '''Attempt to refresh expired token using refresh_token'''
    try:
        # This would implement OAuth 2.1 refresh token flow
        # For services that support it (like Replicate)
        logger.info(f"Token refresh not yet implemented for {service_name}")
        return False
    except Exception as e:
        logger.error(f"Token refresh failed for {service_name}: {e}")
        return False

@dataclass
class TokenValidationResult:
    is_valid: bool
    is_expired: bool  
    needs_refresh: bool
"""

# STEP 4: Improved Error Detection and Reporting

ERROR_DETECTION_ENHANCEMENTS = """
# Enhanced error classification based on MCP spec:

def classify_mcp_error(self, error: Exception, service_name: str) -> dict:
    '''Classify errors according to MCP OAuth 2.1 specifications'''
    error_str = str(error).lower()
    
    # Per MCP spec: "Invalid or expired tokens MUST receive a HTTP 401 response"
    if '401' in error_str or 'unauthorized' in error_str:
        return {
            'error_type': 'token_expired',
            'error_code': 401,
            'description': 'Access token is invalid or expired',
            'action_required': 'refresh_or_reauth',
            'mcp_spec_reference': 'OAuth 2.1 Section 5.3'
        }
    
    # Per MCP spec: "403 Forbidden = Invalid scopes or insufficient permissions"  
    elif '403' in error_str or 'forbidden' in error_str:
        return {
            'error_type': 'insufficient_permissions',
            'error_code': 403, 
            'description': 'Invalid scopes or insufficient permissions',
            'action_required': 'check_scopes',
            'mcp_spec_reference': 'MCP Authorization Spec'
        }
    
    # Per MCP spec: "400 Bad Request = Malformed authorization request"
    elif '400' in error_str or 'bad request' in error_str:
        return {
            'error_type': 'malformed_request',
            'error_code': 400,
            'description': 'Malformed authorization request', 
            'action_required': 'check_request_format',
            'mcp_spec_reference': 'OAuth 2.1 Request Format'
        }
    
    else:
        return {
            'error_type': 'unknown',
            'error_code': None,
            'description': str(error),
            'action_required': 'investigate',
            'mcp_spec_reference': None
        }
"""

print("Integration plan created. Key improvements needed:")
print("1. ✅ Add expires_at timestamp when saving tokens")
print("2. ✅ Validate token expiration before API calls") 
print("3. ✅ Detect and handle 401 responses specifically")
print("4. ✅ Implement refresh token logic (if supported)")
print("5. ✅ Classify errors per MCP/OAuth 2.1 specifications")
print("6. ✅ Clear expired tokens automatically")
print("7. ✅ Provide detailed error reporting with MCP spec references")
