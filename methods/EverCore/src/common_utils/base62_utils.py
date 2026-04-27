"""
Base62 encoding utility
Converts numeric IDs to short strings using 62 characters: 0-9, a-z, A-Z
"""

# Base62 character set: 0-9 (10 characters) + a-z (26 characters) + A-Z (26 characters) = 62 characters
BASE62_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(BASE62_CHARSET)


def encode_base62(num: int) -> str:
    """
    Encode a decimal number into a Base62 string

    Args:
        num: Decimal number to encode (must be >= 0)

    Returns:
        str: Base62 encoded string

    Raises:
        ValueError: When input number is less than 0

    Examples:
        >>> encode_base62(0)
        '0'
        >>> encode_base62(61)
        'Z'
        >>> encode_base62(62)
        '10'
        >>> encode_base62(1000000)
        '4C92'
    """
    if num < 0:
        raise ValueError("Input number must be greater than or equal to 0")

    if num == 0:
        return BASE62_CHARSET[0]

    result = []
    while num > 0:
        result.append(BASE62_CHARSET[num % BASE])
        num //= BASE

    # Reverse the result since we built it from least significant to most significant
    return ''.join(reversed(result))


def decode_base62(encoded: str) -> int:
    """
    Decode a Base62 string into a decimal number

    Args:
        encoded: Base62 encoded string

    Returns:
        int: Decoded decimal number

    Raises:
        ValueError: When the string contains invalid characters

    Examples:
        >>> decode_base62('0')
        0
        >>> decode_base62('Z')
        61
        >>> decode_base62('10')
        62
        >>> decode_base62('4C92')
        1000000
    """
    if not encoded:
        raise ValueError("Encoded string cannot be empty")

    result = 0
    for char in encoded:
        if char not in BASE62_CHARSET:
            raise ValueError(f"Invalid character: {char}")
        result = result * BASE + BASE62_CHARSET.index(char)

    return result


def generate_short_code(id_value: int, min_length: int = 4) -> str:
    """
    Generate a short link code based on ID

    Args:
        id_value: Database ID value
        min_length: Minimum length, pad with leading zeros if shorter (default is 4)

    Returns:
        str: Generated short link code

    Examples:
        >>> generate_short_code(1)
        '0001'
        >>> generate_short_code(62)
        '0010'
        >>> generate_short_code(1000000)
        '4C92'
    """
    if id_value < 0:
        raise ValueError("ID value must be greater than or equal to 0")

    encoded = encode_base62(id_value)

    # If length is less than minimum, pad with leading zeros
    if len(encoded) < min_length:
        encoded = BASE62_CHARSET[0] * (min_length - len(encoded)) + encoded

    return encoded


def is_valid_short_code(short_code: str) -> bool:
    """
    Validate whether a short link code is valid

    Args:
        short_code: Short link code to validate

    Returns:
        bool: Whether the code is valid
    """
    if not short_code:
        return False

    # Check if it contains only characters from the Base62 character set
    return all(char in BASE62_CHARSET for char in short_code)


def extract_id_from_short_code(short_code: str) -> int:
    """
    Extract the original ID from a short link code

    Args:
        short_code: Short link code

    Returns:
        int: Original ID value

    Raises:
        ValueError: When the short link code is invalid
    """
    if not is_valid_short_code(short_code):
        raise ValueError(f"Invalid short link code: {short_code}")

    return decode_base62(short_code)
