# FastMCP OAuth Auto-Refresh: Root Cause Analysis

**Date:** November 4, 2025
**FastMCP Version:** 2.13.0.2 (latest)
**Status:** Issue #1863 still OPEN

## Executive Summary

**FastMCP OAuth auto-refresh does NOT work due to a fundamental architectural bug:**

The entire OAuth token structure (including `refresh_token`) is saved with TTL equal to `access_token` expiry time. When the `access_token` expires, DiskStore deletes the **entire** entry, including the `refresh_token` needed for auto-refresh.

**Impact:** Users must manually re-authorize every time `access_token` expires (~1 hour for most providers).

---

## Root Cause

### Code Location
`/fastmcp/client/auth/oauth.py` lines 107-112:

```python
@override
async def set_tokens(self, tokens: OAuthToken) -> None:
    await self._storage_oauth_token.put(
        key=self._get_token_cache_key(),
        value=tokens,
        ttl=tokens.expires_in,  # ← BUG: TTL applies to ENTIRE token structure
    )
```

### Token Structure
```python
OAuthToken {
    access_token: str       # Expires in ~1 hour
    refresh_token: str      # Long-lived (days/weeks)
    expires_in: int         # Duration of access_token (e.g., 3600)
    token_type: str         # "Bearer"
}
```

### The Bug

1. **Token Saved with TTL:**
   ```python
   # When saving tokens after OAuth flow
   storage.put(key="token", value=entire_token, ttl=3600)  # 1 hour
   ```

2. **DiskStore Behavior:**
   - Stores entry with `expire_time = now() + 3600`
   - After 1 hour: `expire_time < now()`
   - `.get()` returns `None` for expired entries

3. **Token Retrieval Fails:**
   ```python
   # When loading tokens on next run
   tokens = await storage.get("token")  # Returns None (expired!)
   # refresh_token is GONE even though it's still valid
   ```

4. **Auto-Refresh Impossible:**
   ```python
   def can_refresh_token(self) -> bool:
       return bool(self.current_tokens and  # ← None!
                   self.current_tokens.refresh_token and
                   self.client_info)
   # Returns False → browser opens
   ```

### Evidence

**Test Results:**
```bash
$ python test_expiry_bug.py

1. Database expire_time: 1762260256 (1 hour ago)
2. After FastMCP _initialize():
   context.token_expiry_time: None
   context.is_token_valid(): False
   context.can_refresh_token(): False  # ← Cannot refresh!

$ python -c "await DiskStore.get('expired_token')"
# Returns: None  ← Entry deleted by TTL
```

---

## Why Auto-Refresh Can Never Work

### OAuth Flow (Current Implementation)

```
Time 0: Initial OAuth
├─ User authorizes via browser
├─ Receive: access_token (expires 1h) + refresh_token (expires 30d)
└─ Save: BOTH tokens with TTL=3600 (1 hour)

Time 1h: Access token expires
├─ DiskStore: expire_time < now() → DELETE ENTIRE ENTRY
├─ FastMCP: storage.get() → None
├─ No refresh_token available
└─ Browser opens for full re-authorization
```

### OAuth Flow (How It Should Work)

```
Time 0: Initial OAuth
├─ User authorizes via browser
├─ Receive: access_token (expires 1h) + refresh_token (expires 30d)
└─ Save: BOTH tokens with TTL=None OR TTL=refresh_token_expiry

Time 1h: Access token expires
├─ FastMCP: storage.get() → returns tokens (NOT deleted)
├─ Check: access_token expired, refresh_token valid
├─ Use refresh_token to get new access_token
└─ Save new access_token, keep refresh_token
```

---

## Related Issues

### GitHub Issue #1863
**Title:** "OAuth Token Refresh Does Not Update Auth Context"
**Status:** OPEN (last updated October 29, 2025)

**Description:** Even if refresh works, new token doesn't update `auth_context_var` (Python `ContextVar` immutability issue).

**Our Finding:** Issue #1863 is a **secondary** problem. The **primary** problem (this one) prevents refresh from even being attempted!

### GitHub Issue #1275
**Title:** "Internal Server Error when access token expires"
**Status:** Reported July 2025

Users getting 500 errors when trying to use refresh tokens.

---

## Solutions

### Option 1: Remove TTL for OAuth Tokens (Recommended)
```python
async def set_tokens(self, tokens: OAuthToken) -> None:
    await self._storage_oauth_token.put(
        key=self._get_token_cache_key(),
        value=tokens,
        ttl=None,  # ← Never expire in storage
    )
    # Let FastMCP logic handle expiry checks, not DiskStore
```

**Pros:**
- Simple fix
- refresh_token always available
- FastMCP can check `is_token_valid()` in-memory

**Cons:**
- Tokens persist forever (minor issue)

### Option 2: Use Refresh Token Expiry for TTL
```python
async def set_tokens(self, tokens: OAuthToken) -> None:
    # Calculate TTL based on refresh_token lifetime (if available)
    ttl = tokens.refresh_token_expires_in if hasattr(tokens, 'refresh_token_expires_in') else None

    await self._storage_oauth_token.put(
        key=self._get_token_cache_key(),
        value=tokens,
        ttl=ttl,  # ← Use refresh_token expiry, not access_token
    )
```

**Pros:**
- Tokens eventually expire (security)
- refresh_token available during its lifetime

**Cons:**
- refresh_token expiry not always in token response
- More complex logic

### Option 3: Separate Storage for Refresh Tokens
```python
# Store access_token with short TTL
await self._storage_access_token.put(
    key="access_token",
    value=tokens.access_token,
    ttl=tokens.expires_in
)

# Store refresh_token separately with no TTL
await self._storage_refresh_token.put(
    key="refresh_token",
    value=tokens.refresh_token,
    ttl=None
)
```

**Pros:**
- Clean separation
- Correct lifetime management

**Cons:**
- Requires refactoring
- Two storage operations

---

## Why Token Persistence "Works" in Our Tests

### Run #1: Fresh OAuth (21s)
- Browser opens → user authorizes
- Tokens saved with TTL=3600
- Immediately used → SUCCESS

### Run #2: Valid Token (7s)
- Load tokens from storage
- expire_time = saved_time + 3600
- Still within 1 hour → tokens returned
- SUCCESS (no browser)

### Run #3: Expired Token (18s + browser)
- Load tokens from storage
- expire_time < now() → **DiskStore returns None**
- No refresh_token available
- Browser opens → FAIL

**Misleading Result:** Runs #1 and #2 made it look like "token persistence works!" But it only works for **valid** tokens, not expired ones needing refresh.

---

## FastMCP Release Analysis

### v2.13.0.2 (October 28, 2025) - LATEST
**OAuth Changes:**
- "Check if refresh_token is None" (PR #2025)
- OAuth proxy improvements
- Token introspection support

**Our Finding:** The PR #2025 checks if `refresh_token` is None, but doesn't fix the TTL issue. Tokens are still deleted on expiry.

### Issue #1863 Status
- **Opened:** September 19, 2025
- **Last Updated:** October 29, 2025
- **Status:** OPEN
- **Comments:** 6 interactions, no merged fix

**Conclusion:** FastMCP team is aware of token refresh issues but has not fixed the root cause (TTL deletion).

---

## Testing Methodology

### What We Tested

1. **Token Persistence** ✅
   - Fresh OAuth flow
   - Token storage in DiskStore
   - Token loading on subsequent runs

2. **Token Reuse** ✅
   - Valid tokens loaded and used
   - No browser window on repeat runs
   - 64-93% performance improvement

3. **Auto-Refresh** ❌
   - Manually expired token in database
   - Attempted to use refresh_token
   - **FAILED:** DiskStore returned None

### Critical Test
```python
# 1. Expire token in database
db.execute("UPDATE Cache SET expire_time = ? WHERE key = ?",
           (time.time() - 3600, token_key))

# 2. Try to load expired token
token = await storage.get(token_key)
# Result: None (expected: token with refresh_token)

# 3. FastMCP cannot refresh
oauth.context.can_refresh_token()  # False
# Result: Browser opens
```

---

## Recommendations

### For Production Use

**Workaround #1: Proactive Re-authorization**
```python
# Check expiry before it happens
if time.time() > (token_expire_time - 300):  # 5 min buffer
    # Trigger re-auth before expiry
    await trigger_oauth_flow()
```

**Workaround #2: Extended Sessions**
- Configure OAuth provider for longer token lifetimes (if supported)
- Some providers allow 8-24 hour access tokens

**Workaround #3: Manual Token Management**
- Don't use FastMCP's OAuth for production
- Implement custom OAuth with correct TTL handling

### For FastMCP Development

**Submit Bug Report:**
1. Reference this analysis
2. Link to Issue #1863
3. Propose Option #1 fix (remove TTL)
4. Include test reproduction steps

**Pull Request:**
```python
# fastmcp/client/auth/oauth.py
async def set_tokens(self, tokens: OAuthToken) -> None:
    await self._storage_oauth_token.put(
        key=self._get_token_cache_key(),
        value=tokens,
        ttl=None,  # FIX: Don't use access_token TTL for storage
    )
```

---

## Impact Assessment

| Scenario | Current Behavior | Desired Behavior |
|----------|-----------------|------------------|
| **Fresh OAuth** | ✅ Works (21s) | ✅ Works |
| **Valid token** | ✅ Works (7s) | ✅ Works |
| **Expired token** | ❌ Browser opens (18s) | ✅ Auto-refresh (2-3s) |
| **Production 24h** | ❌ 24 re-authorizations | ✅ 1 OAuth, 23 refreshes |

**Performance Impact:**
- Current: User must authorize every hour
- Fixed: User authorizes once, silent refresh thereafter
- Time saved: 15-18s per refresh (no browser interaction)

**Security Impact:**
- Current: Less secure (repeated browser flows)
- Fixed: More secure (fewer auth flows, refresh token rotation)

---

## Conclusion

**FastMCP OAuth auto-refresh is fundamentally broken due to incorrect TTL usage.**

The bug is in `oauth.py:111` where the entire token structure (including long-lived `refresh_token`) is saved with TTL equal to short-lived `access_token` expiry.

**Status:**
- ✅ Token persistence works (for valid tokens)
- ✅ Token loading works (for valid tokens)
- ❌ Auto-refresh does NOT work (tokens deleted before refresh)
- ❌ Issue #1863 still OPEN

**Fix Required:** Remove `ttl` parameter or use `refresh_token` lifetime instead of `access_token` lifetime.

**Workaround:** Manually re-authorize every hour, or don't rely on FastMCP OAuth for production.
