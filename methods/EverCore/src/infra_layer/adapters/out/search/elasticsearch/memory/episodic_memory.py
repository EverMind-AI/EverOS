# Import retained for type annotations and field definitions
from elasticsearch.dsl import field as e_field
from core.tenants.tenantize.oxm.es.tenant_aware_async_document import (
    TenantAwareAliasDoc,
)
from core.oxm.es.analyzer import whitespace_lowercase_trim_stop_analyzer


class EpisodicMemoryDoc(
    TenantAwareAliasDoc("v1_episodic_memory", number_of_shards=3, number_of_replicas=1)
):
    """
    V1 Episodic Memory Elasticsearch Document

    Based on MongoDB v1_episodic_memories collection.
    Simplified for BM25 text retrieval - only stores search-essential fields.
    Full data is retrieved from MongoDB using parent_id.

    Field descriptions:
    - id: Event unique identifier (corresponds to MongoDB _id)
    - user_id: User ID (optional, None for group memory)
    - group_id: Group ID (optional)
    - session_id: Session identifier (optional)
    - timestamp: Event occurrence time
    - participants: List of participant sender_ids
    - sender_ids: Sender IDs (multi-value)
    - summary: Event summary
    - subject: Event subject/title
    - episode: Full episodic description (core BM25 content)
    - search_content: BM25 search field (supports multi-value storage, used for exact word matching)
    - type: Event type (Conversation, etc.)
    - parent_type: Parent memory type (e.g., memcell)
    - parent_id: Parent memory ID (for MongoDB back-reference)

    Tokenization notes:
    - Application layer is responsible for Chinese tokenization (jieba recommended)
    - subject, episode, and summary fields store pre-tokenized results (space-separated)
    - search_content field supports multi-value storage, each value being a search term
    - ES uses whitespace_lowercase_trim_stop analyzer for text fields
    """

    class CustomMeta:
        # Specify the field name used to automatically populate meta.id
        id_source_field = "id"

    # Basic identifier fields
    id = e_field.Keyword(required=True)
    user_id = e_field.Keyword()  # None for group memory
    group_id = e_field.Keyword()
    session_id = e_field.Keyword()

    # Timestamp field
    timestamp = e_field.Date(required=True)

    # Participant list
    participants = e_field.Keyword(multi=True)
    sender_ids = e_field.Keyword(multi=True)

    # Core BM25 content fields
    summary = e_field.Text(
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )

    subject = e_field.Text(
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )

    episode = e_field.Text(
        required=True,
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )

    search_content = e_field.Text(
        multi=True,
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )

    # Classification fields
    type = e_field.Keyword()  # Conversation/Email/Notion, etc.

    # Parent info for MongoDB back-reference
    parent_type = e_field.Keyword()
    parent_id = e_field.Keyword()
