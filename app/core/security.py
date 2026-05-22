"""
API key authentication and in-memory rate limiting.
No Redis dependency — rate limiting uses a simple in-process counter.
"""
import hashlib
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    Validate the API key from the X-API-Key header.
    Returns the key if valid, raises HTTP 401 otherwise.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not _constant_time_compare(api_key, settings.api_key):
        logger.warning(
            "invalid_api_key_attempt",
            ip=request.client.host if request.client else "unknown",
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def _constant_time_compare(val1: str, val2: str) -> bool:
    """Constant-time string comparison — prevents timing attacks."""
    if len(val1) != len(val2):
        return False
    result = 0
    for x, y in zip(val1.encode(), val2.encode()):
        result |= x ^ y
    return result == 0


# ── In-process sliding-window rate limiter ────────────────────────────────────
# Stores per-key request timestamps in memory.
# Resets on process restart — sufficient for single-instance deployments.
# For multi-replica deployments, replace with a Redis-backed implementation.

_rate_limit_store: Dict[str, List[float]] = defaultdict(list)


def check_rate_limit(identifier: str) -> None:
    """
    Raise HTTP 429 if the identifier has exceeded the configured rate limit.
    Uses a sliding window of settings.rate_limit_window seconds.
    """
    key       = hashlib.md5(identifier.encode()).hexdigest()
    now       = time.time()
    window    = settings.rate_limit_window
    max_reqs  = settings.rate_limit_requests

    # Prune timestamps outside the current window
    timestamps = _rate_limit_store[key]
    _rate_limit_store[key] = [t for t in timestamps if t > now - window]

    if len(_rate_limit_store[key]) >= max_reqs:
        reset_at = int(min(_rate_limit_store[key]) + window)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {max_reqs} requests per {window}s.",
            headers={
                "X-RateLimit-Limit":     str(max_reqs),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset":     str(reset_at),
                "Retry-After":           str(window),
            },
        )

    _rate_limit_store[key].append(now)
