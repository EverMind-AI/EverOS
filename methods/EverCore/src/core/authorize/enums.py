from enum import Enum


class Role(Enum):
    """User role enumeration"""

    ANONYMOUS = "anonymous"  # Anonymous user
    USER = "user"  # Regular user
    ADMIN = "admin"  # Administrator
    SIGNATURE = "signature"  # HMAC signature verification user
