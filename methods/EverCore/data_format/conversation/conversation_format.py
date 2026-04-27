"""
Open source conversation format definition

This module defines a standard conversation data format for storing and exchanging conversation data.
The format design considers extensibility, readability, and data completeness.
"""

from typing import TypedDict, List, Optional, Literal, Dict, Any


# Role type for identifying message/user source (human or AI)
# Compatible with OpenAI/mem0/memos message format
RoleType = Literal["user", "assistant"]


# User detailed information
class UserDetail(TypedDict, total=False):
    """User detailed information

    Attributes:
        full_name: User full name (optional)
        role: User type role ("user" for human, "assistant" for AI)
        custom_role: User's job/position role (e.g. product manager, technical lead, etc.)
        email: Email address (optional)
        avatar_url: Avatar URL (optional)
        department: Department (optional)
        extra: Other extended information (optional)
    """

    full_name: Optional[str]
    role: Optional[RoleType]
    custom_role: Optional[str]
    email: Optional[str]
    avatar_url: Optional[str]
    department: Optional[str]
    extra: Optional[Dict[str, Any]]


# Session meta information
class SessionMeta(TypedDict, total=False):
    """Session meta information

    Attributes:
        scene: Scene type, supports "solo" (1 user + N agents) or "team" (multi-user + agents) (optional)
        scene_desc: Scene description information (optional)
        name: Group chat name
        description: Group chat description
        group_id: Group chat unique identifier (optional)
        created_at: Group chat creation time (optional)
        default_timezone: Default timezone, if the message has no timezone information, use this timezone (optional)
        user_details: User detailed information dictionary, key is user ID
        tags: Tag list (optional)
        extra: Other extended information (optional)
    """

    scene: Optional[str]
    scene_desc: Optional[Dict[str, Any]]
    name: str
    description: str
    group_id: Optional[str]
    created_at: Optional[str]
    default_timezone: Optional[str]
    user_details: Dict[str, UserDetail]
    tags: Optional[List[str]]
    extra: Optional[Dict[str, Any]]


# Message type
MessageType = Literal["text", "image", "file", "audio", "video", "link", "system"]


# Message reference object (only message_id is required)
class MessageReference(TypedDict, total=False):
    """Message reference object

    When using object form in refer_list, only message_id is required,
    other fields are optional, and can be added flexibly as needed.

    Attributes:
        message_id: The ID of the referenced message (required)
        create_time: Message creation time (optional)
        sender: Sender user ID (optional)
        sender_name: Sender name (optional)
        type: Message type (optional)
        content: Message content (optional)
        refer_list: Referenced message list (optional, supports nested references)
        extra: Other extended information (optional)
    """

    message_id: str  # Unique required field
    create_time: Optional[str]
    sender: Optional[str]
    sender_name: Optional[str]
    type: Optional[MessageType]
    content: Optional[str]
    refer_list: Optional[List[Any]]  # Supports nested references
    extra: Optional[Dict[str, Any]]


# Mark message_id as required field
MessageReference.__required_keys__ = frozenset(["message_id"])


# Message
class Message(TypedDict, total=False):
    """Single message

    Attributes:
        message_id: Message unique identifier
        create_time: Message creation time (ISO 8601 format, it is recommended to include timezone information)
        sender: Sender user ID
        sender_name: Sender name (optional, for quick view, detailed information in user_details)
        role: Message sender role (optional), used to identify the source of the message.
              "user" for human messages, "assistant" for AI messages.
              Compatible with OpenAI/mem0/memos message format.
        type: Message type (text/image/file etc.)
        content: Message content, according to type different may be text, file URL etc.
        refer_list: Referenced message list (optional)
                    Supports two formats:
                    1. String: Directly use message_id, such as ["msg_001", "msg_002"]
                    2. Object: MessageReference object, only message_id is required, other fields are optional
                      e.g. [{"message_id": "msg_001", "content": "Quote content"}]
        extra: Other extended information, such as emoji reply, edit history etc. (optional)
    """

    message_id: str
    create_time: str
    sender: str
    sender_name: Optional[str]
    role: Optional[RoleType]
    type: MessageType
    content: str
    refer_list: Optional[List[Any]]  # Can be MessageReference or str (message_id)
    extra: Optional[Dict[str, Any]]


# Complete conversation format
class ConversationFormat(TypedDict):
    """Complete conversation format

    Attributes:
        version: Format version number (following semantic version)
        session_meta: Session meta information
        conversation_list: Message list
    """

    version: str
    session_meta: SessionMeta
    conversation_list: List[Message]


def validate_conversation_format(data: ConversationFormat) -> bool:
    """Validate conversation format whether it conforms to the specification

    Args:
        data: Conversation data

    Returns:
        Whether it conforms to the specification
    """
    # Basic field check
    if (
        "version" not in data
        or "session_meta" not in data
        or "conversation_list" not in data
    ):
        return False

    meta = data["session_meta"]
    if "name" not in meta or "user_details" not in meta:
        return False

    # Check if the sender in the message is in user_details
    user_ids = set(meta["user_details"].keys())
    message_ids = set()

    for msg in data["conversation_list"]:
        # Check sender
        if msg.get("sender") not in user_ids:
            return False

        # Collect all message_id
        if "message_id" in msg:
            message_ids.add(msg["message_id"])

    # Check references in refer_list
    for msg in data["conversation_list"]:
        refer_list = msg.get("refer_list", [])
        if refer_list:
            for refer in refer_list:
                # If it is a string (message ID reference)
                if isinstance(refer, str):
                    # Can validate if the ID exists in conversation_list
                    # if refer not in message_ids:
                    #     return False
                    pass
                # If it is a dictionary (MessageReference object)
                elif isinstance(refer, dict):
                    # Only validate required fields: message_id
                    if "message_id" not in refer:
                        return False
                    # If it contains the sender field, validate if it is in user_details
                    if "sender" in refer and refer.get("sender") not in user_ids:
                        return False
                else:
                    return False

    return True


def create_example_conversation() -> ConversationFormat:
    """Create an example conversation data

    Returns:
        Example conversation data
    """
    return {
        "version": "1.0.0",
        "session_meta": {
            "name": "Smart Sales Assistant Project Group",
            "description": "Smart Sales Assistant Project Development Discussion Group",
            "group_id": "group_sales_ai_2025",
            "created_at": "2025-02-01T01:00:00Z",
            "default_timezone": "UTC",
            "user_details": {
                "user_101": {
                    "full_name": "Alex",
                    "role": "user",
                    "custom_role": "Technical Lead",
                    "department": "Technology Department",
                },
                "user_102": {
                    "full_name": "Betty",
                    "role": "user",
                    "custom_role": "Product Manager",
                    "department": "Product Department",
                },
                "user_103": {
                    "full_name": "Chen",
                    "role": "user",
                    "custom_role": "Project Manager",
                    "department": "Project Management Department",
                },
                "user_104": {
                    "full_name": "Dylan",
                    "role": "user",
                    "custom_role": "Backend Engineer",
                    "department": "Technology Department",
                },
                "user_105": {
                    "full_name": "Emily",
                    "role": "user",
                    "custom_role": "Frontend Engineer",
                    "department": "Technology Department",
                },
            },
            "tags": ["AI", "Sales", "Project Development"],
        },
        "conversation_list": [
            {
                "message_id": "msg_001",
                "create_time": "2025-02-01T02:00:00Z",
                "sender": "user_103",
                "sender_name": "Chen",
                "type": "text",
                "content": "Good morning, \"Smart Sales Assistant\" now how is the progress?",
                "refer_list": [],
            },
            {
                "message_id": "msg_002",
                "create_time": "2025-02-01T02:01:00Z",
                "sender": "user_102",
                "sender_name": "Betty",
                "type": "text",
                "content": "Good morning. First align the goals? Is it an MVP that can be used for internal testing, or a direct customer trial?",
                "refer_list": [],
            },
            {
                "message_id": "msg_003",
                "create_time": "2025-02-01T02:01:30Z",
                "sender": "user_103",
                "sender_name": "Chen",
                "type": "text",
                "content": "First the MVP, have something running by March, and wrap up by the end of April.",
                # Method 1: Only reference message ID (simple reference)
                "refer_list": ["msg_002"],
            },
            {
                "message_id": "msg_004",
                "create_time": "2025-02-01T02:02:00Z",
                "sender": "user_101",
                "sender_name": "Alex",
                "type": "text",
                "content": "Technical advice mainly based on RAG.",
                "refer_list": [],
            },
            {
                "message_id": "msg_005",
                "create_time": "2025-02-01T02:02:30Z",
                "sender": "user_101",
                "sender_name": "Alex",
                "type": "text",
                "content": "Current method: BM25(ES)+vector retrieval(bge-base-zh, HNSW, topK=8)+cross-encoder re-ranking(starting with the base version), temperature 0.3, length 512.",
                "refer_list": [],
            },
            {
                "message_id": "msg_006",
                "create_time": "2025-02-01T02:03:30Z",
                "sender": "user_102",
                "sender_name": "Betty",
                "type": "text",
                "content": "First define the metrics: - effective hit rate≥0.8; - first response P95≤1.5s; - hallucination rate≤8%.",
                # Method 2: Object reference, can only contain message_id
                "refer_list": [{"message_id": "msg_004"}],
            },
            {
                "message_id": "msg_007",
                "create_time": "2025-02-01T02:04:00Z",
                "sender": "user_101",
                "sender_name": "Alex",
                "type": "text",
                "content": "Source data: product manual v3.2, price policy Q1, delivery SLA document.",
                # Method 3: Object reference, can optionally include some fields (such as content for preview)
                "refer_list": [
                    {
                        "message_id": "msg_006",
                        "content": "First define the metrics: - effective hit rate≥0.8; - first response P95≤1.5s; - hallucination rate≤8%.",
                    }
                ],
            },
            {
                "message_id": "msg_008",
                "create_time": "2025-02-01T02:05:00Z",
                "sender": "user_101",
                "sender_name": "Alex",
                "type": "text",
                "content": "I have considered multiple technical points.",
                # Method 4: Mixed use of string and object
                "refer_list": [
                    "msg_004",  # Simple reference
                    {
                        "message_id": "msg_005",
                        "sender": "user_101",
                        "content": "Current method: BM25(ES)+vector retrieval...",
                    },  # Reference with partial information
                ],
            },
        ],
    }


if __name__ == "__main__":
    import json

    # Create example data
    example = create_example_conversation()

    # Validate format
    is_valid = validate_conversation_format(example)
    print(f"Format validation: {'passed' if is_valid else 'failed'}")

    # Output example JSON
    print("\nExample JSON:")
    print(json.dumps(example, ensure_ascii=False, indent=2))
