import hmac
import hashlib
import time
import os
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.authorize.enums import Role
from core.context.context import set_current_user_info, clear_current_user_context
from core.observation.logger import get_logger
from core.component.redis_provider import RedisProvider
from core.di import get_bean_by_type

logger = get_logger(__name__)


class HMACSignatureMiddleware(BaseHTTPMiddleware):
    """
    HMAC signature verification middleware

    Verifies the HMAC signature of requests to ensure request integrity and authenticity.
    Uses HTTP method, URL path, and timestamp as signing data.
    The time window is 5 minutes; requests exceeding this window will be rejected.
    """

    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        time_window_minutes: int = 5,
        redis_provider: Optional[RedisProvider] = None,
    ):
        """
        Initialize HMAC signature middleware

        Args:
            app: ASGI application instance
            secret_key: Secret key for HMAC signature
            time_window_minutes: Time window (in minutes), default is 5 minutes
            redis_provider: Redis provider, used for replay attack prevention
        """
        super().__init__(app)
        self.secret_key = secret_key.encode('utf-8')
        self.time_window_seconds = time_window_minutes * 60
        self._redis_provider = redis_provider

    @property
    def redis_provider(self) -> RedisProvider:
        return self._redis_provider or get_bean_by_type(RedisProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle HMAC signature verification and set user context

        Expected request headers:
        - X-Timestamp: Unix timestamp (seconds)
        - X-Nonce: Nonce (for replay attack prevention)
        - X-Signature: HMAC-SHA256 signature (hexadecimal format)

        Signature data format: {METHOD}|{URL_PATH}|{TIMESTAMP}|{NONCE}

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response: Response object
        """
        # Clear any existing user context
        clear_current_user_context()

        # Set user context token
        token = None

        # Step 1: Attempt HMAC signature verification and set user context
        try:
            # Get timestamp, nonce, and signature from request headers
            timestamp_header = request.headers.get("X-Timestamp")
            nonce_header = request.headers.get("X-Nonce")
            signature_header = request.headers.get("X-Signature")

            # If there are signature-related headers, perform verification
            if (
                timestamp_header and nonce_header and signature_header
            ) or signature_header == "1234567890":
                # Verify HMAC signature
                is_valid_signature = await self._verify_hmac_signature(
                    request, timestamp_header, nonce_header, signature_header
                )

                if is_valid_signature:
                    # Signature verification succeeded, set user context with SIGNATURE role
                    user_data = {"user_id": -1, "role": Role.SIGNATURE.value}
                    token = set_current_user_info(user_data)
                    logger.info("HMAC signature user context set: role=SIGNATURE")
                else:
                    logger.info(
                        "HMAC signature verification failed, user context not set"
                    )
            else:
                logger.info(
                    "HMAC signature headers not found, skipping signature verification"
                )

        except Exception as e:
            logger.error(
                "Exception during HMAC signature verification: %s, "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                str(e),
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            # Signature verification failure does not affect request processing
            # Specific permission checks are handled by individual endpoints

        # Step 2: Execute business logic
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"Exception in business logic processing: {str(e)}")
            # Re-raise business logic exceptions for upstream handling
            raise

        finally:
            # Clean up user context
            if token is not None:
                try:
                    clear_current_user_context(token)
                    logger.debug("HMAC signature user context cleaned up")
                except Exception as reset_error:
                    logger.warning(
                        f"Error occurred while cleaning up HMAC signature user context: {str(reset_error)}"
                    )

    async def _verify_hmac_signature(
        self,
        request: Request,
        timestamp_header: str,
        nonce_header: str,
        signature_header: str,
    ) -> bool:
        """
        Verify HMAC signature

        Args:
            request: FastAPI request object
            timestamp_header: Timestamp header value
            nonce_header: Nonce header value
            signature_header: Signature header value

        Returns:
            bool: Whether the signature is valid
        """

        if signature_header == "1234567890" and os.getenv("ENV") == "dev":
            return True

        try:
            # Parse timestamp
            request_timestamp = int(timestamp_header)
        except ValueError:
            logger.error(
                "HMAC signature verification failed - invalid timestamp format: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            return False

        # Validate nonce is not empty
        if not nonce_header or not nonce_header.strip():
            logger.error(
                "HMAC signature verification failed - X-Nonce header is empty or invalid: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            return False

        # Replay attack prevention: Use atomic operation to check and store nonce
        nonce_key = f"nonce:{nonce_header}"
        expire_seconds = self.time_window_seconds * 2

        if self.redis_provider:
            try:
                # Use Redis SET NX EX command for atomic operation: set if key does not exist, otherwise return False
                # This completes the check and storage in one atomic operation, avoiding race conditions
                nonce_stored = await self.redis_provider.set(
                    nonce_key, str(request_timestamp), ex=expire_seconds, nx=True
                )
                if not nonce_stored:
                    logger.error(
                        "HMAC signature verification failed - replay attack detected, nonce has been used: "
                        "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                        "method='%s', url_path='%s', request_timestamp=%d, current_time=%d",
                        timestamp_header,
                        nonce_header,
                        signature_header,
                        request.method,
                        request.url.path,
                        request_timestamp,
                        int(time.time()),
                    )
                    return False
                logger.debug(
                    "Nonce stored in Redis: %s, expiration time=%d seconds",
                    nonce_header,
                    expire_seconds,
                )
            except Exception as e:
                logger.error(
                    "Error occurred while checking and storing nonce: %s", str(e)
                )
                # If Redis is unavailable, log a warning but do not block the request (degraded mode)
                logger.warning("Redis unavailable, skipping nonce replay check")

        # Validate time window
        current_time = int(time.time())
        time_diff = abs(current_time - request_timestamp)

        if time_diff > self.time_window_seconds:
            logger.error(
                "HMAC signature verification failed - request exceeds time window: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s', current_time=%d, request_timestamp=%d, "
                "time_diff=%d seconds, time_window=%d seconds",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
                current_time,
                request_timestamp,
                time_diff,
                self.time_window_seconds,
            )
            return False

        # Build signature data
        method = request.method
        url_path = request.url.path
        signature_data = f"{method}|{url_path}|{request_timestamp}|{nonce_header}"

        # Calculate expected signature
        expected_signature = hmac.new(
            self.secret_key, signature_data.encode('utf-8'), hashlib.sha256
        ).hexdigest()

        # Verify signature
        if not hmac.compare_digest(signature_header, expected_signature):
            logger.error(
                "HMAC signature verification failed - signature mismatch: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s', signature_data='%s', "
                "expected_signature='%s', secret_key_length=%d",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
                signature_data,
                expected_signature,
                len(self.secret_key),
            )
            return False

        logger.debug("HMAC signature verification succeeded: %s %s", method, url_path)
        return True


def get_hmac_security_config():
    """
    Get OpenAPI security configuration for HMAC signature authentication

    Returns:
        List[dict]: List of OpenAPI security configurations defining requirements for HMAC signature authentication
    """
    return [{"HMACSignature": []}]


def get_hmac_openapi_security_schemes():
    """
    Get OpenAPI security scheme definitions for HMAC signature authentication

    This function returns the configuration used in OpenAPI components.securitySchemes,
    defining the specific implementation and parameters of HMAC signature authentication.

    Returns:
        dict: OpenAPI securitySchemes configuration
    """
    return {
        "HMACSignature": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Signature",
            "description": """**HMAC Signature Authentication**

Uses the HMAC-SHA256 algorithm to sign and verify requests, ensuring request integrity and authenticity.

**Signing Algorithm:**
1. Construct signature data: `{HTTP_METHOD}|{URL_PATH}|{TIMESTAMP}|{NONCE}`
   - Example: `POST|/finance/storage/sign/download|1755572417|7fe6a3edabb9c1383b6d75a72ffce2e5`
2. Compute signature using HMAC-SHA256 algorithm and shared secret key
3. Convert signature to hexadecimal string

**Required Request Headers:**
- `X-Timestamp`: Unix timestamp (seconds), used for replay attack prevention
- `X-Nonce`: Nonce (used for replay attack prevention, must be unique per request)
- `X-Signature`: HMAC-SHA256 signature (hexadecimal format)

**Replay Attack Prevention Mechanism:**
- The difference between request timestamp and server time must not exceed 5 minutes
- Each nonce can only be used once; the server records used nonces in Redis
- The nonce expiration time in Redis is twice the time window (10 minutes), ensuring security
- Requests reusing the same nonce will be rejected

**Signature Example:**
```bash
# Generate signature using HMAC signature generator tool
python tests/hmac_signature_generator.py -m POST -p "/finance/storage/sign/download" -k "abc-12345"

# Or generate signature manually (assuming key is "abc-12345")
TIMESTAMP=$(date +%s)
NONCE=$(openssl rand -hex 16)
SIGNATURE_DATA="POST|/finance/storage/sign/download|${TIMESTAMP}|${NONCE}"
SIGNATURE=$(echo -n "$SIGNATURE_DATA" | openssl dgst -sha256 -hmac "abc-12345" | cut -d' ' -f2)

curl -X POST "https://api.example.com/finance/storage/sign/download" \\
     -H "X-Timestamp: ${TIMESTAMP}" \\
     -H "X-Nonce: ${NONCE}" \\
     -H "X-Signature: ${SIGNATURE}"
```

**Python Code Example:**
```python
import hmac
import hashlib
import secrets
import time

# Complete example of generating a signature
method = "POST"
url_path = "/finance/storage/sign/download"
timestamp = int(time.time())
nonce = secrets.token_hex(16)
secret_key = "abc-12345"

# Construct signature data
signature_data = f"{method}|{url_path}|{timestamp}|{nonce}"
signature = hmac.new(
    secret_key.encode('utf-8'),
    signature_data.encode('utf-8'),
    hashlib.sha256
).hexdigest()

# Request headers
headers = {
    "X-Timestamp": str(timestamp),
    "X-Nonce": nonce,
    "X-Signature": signature
}
```

**Development Environment Shortcut:**
- When environment variable `ENV=dev`, you can use `X-Signature: 1234567890` as a test signature
- Production environment must use correct HMAC signature

**Security Notes:**
- The secret key must be kept confidential and not leaked
- It is recommended to rotate keys regularly
- Ensure client clock is synchronized with server clock
- Nonce should be generated using a cryptographically secure random number generator
- Redis is used to store nonces; ensure Redis security and availability""",
            "x-example": {
                "X-Timestamp": "1755572417",
                "X-Nonce": "7fe6a3edabb9c1383b6d75a72ffce2e5",
                "X-Signature": "6c17b2d568d42b9e0a9df422133f3e84bf4c3aa9bed04400843822586f25e4cd",
            },
        }
    }


def create_hmac_middleware(
    secret_key: str,
    time_window_minutes: int = 5,
    redis_provider: Optional[RedisProvider] = None,
):
    """
    Factory function to create HMAC signature middleware

    Args:
        secret_key: Secret key for HMAC signature
        time_window_minutes: Time window (in minutes), default is 5 minutes
        redis_provider: Redis provider, used for replay attack prevention

    Returns:
        Callable: Middleware constructor
    """

    def middleware_factory(app: ASGIApp) -> HMACSignatureMiddleware:
        return HMACSignatureMiddleware(
            app, secret_key, time_window_minutes, redis_provider
        )

    return middleware_factory
