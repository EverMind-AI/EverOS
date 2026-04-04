"""
Episodic Memory Elasticsearch Repository

A specialized repository class for episodic memory based on BaseRepository, providing efficient BM25 text retrieval and complex query capabilities.
Main features include multi-word search, filtered queries, and document management.
"""

from datetime import datetime
import pprint
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
    EpisodicMemoryDoc,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from common_utils.text_utils import SmartTextParser
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("episodic_memory_es_repository", primary=True)
class EpisodicMemoryEsRepository(BaseRepository[EpisodicMemoryDoc]):
    """
    Episodic Memory Elasticsearch Repository

    A specialized repository class based on BaseRepository, providing:
    - Efficient BM25 text retrieval
    - Multi-word queries and filtering capabilities
    - Document creation and management
    - Manual index refresh control
    """

    def __init__(self):
        """Initialize episodic memory repository"""
        super().__init__(EpisodicMemoryDoc)
        # Initialize smart text parser for calculating intelligent length of query terms
        self._text_parser = SmartTextParser()

    def _calculate_text_score(self, text: str) -> float:
        """
        Calculate intelligent score of text

        Use SmartTextParser to compute the total score of the text, considering weights for different types such as CJK characters, English words, etc.

        Args:
            text: Text to calculate score for

        Returns:
            float: Intelligent score of the text
        """
        if not text:
            return 0.0

        try:
            tokens = self._text_parser.parse_tokens(text)
            return self._text_parser.calculate_total_score(tokens)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "Failed to calculate text score, using character length as fallback: %s",
                e,
            )
            return float(len(text))

    def _log_explanation_details(
        self, explanation: Dict[str, Any], indent: int = 0
    ) -> None:
        """
        Recursively output detailed explanation information

        Args:
            explanation: Explanation dictionary
            indent: Indentation level
        """
        pprint.pprint(explanation, indent=indent)

    # ==================== Document creation and management ====================

    async def create_and_save_episodic_memory(
        self,
        event_id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        user_name: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        linked_entities: Optional[List[str]] = None,
        subject: Optional[str] = None,
        memcell_event_id_list: Optional[List[str]] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        extend: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> EpisodicMemoryDoc:
        """
        Create and save episodic memory document

        Args:
            event_id: Unique event identifier
            user_id: User ID (required)
            timestamp: Event occurrence time (required)
            episode: Episode description (required)
            search_content: List of search content (supports multiple search terms, required)
            type: Event type
            user_name: User name
            title: Event title
            summary: Event summary
            group_id: Group ID
            participants: List of participants
            event_type: Event type
            keywords: List of keywords
            linked_entities: List of linked entity IDs
            subject: Event title (new field)
            memcell_event_id_list: List of memory cell event IDs (new field)
            extend: Extension fields
            created_at: Creation time
            updated_at: Update time

        Returns:
            Saved EpisodicMemoryDoc instance
        """
        try:
            # Set default timestamps
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # Create document instance
            normalized_user_id = user_id or ""
            doc = EpisodicMemoryDoc(
                event_id=event_id,
                type=event_type,
                user_id=normalized_user_id,
                user_name=user_name or '',
                timestamp=timestamp,
                title=title or '',
                episode=episode,
                search_content=search_content,
                summary=summary or '',
                group_id=group_id,
                participants=participants or [],
                keywords=keywords or [],
                linked_entities=linked_entities or [],
                subject=subject or '',
                memcell_event_id_list=memcell_event_id_list or [],
                parent_type=parent_type or '',
                parent_id=parent_id or '',
                extend=extend or {},
                created_at=created_at,
                updated_at=updated_at,
            )

            # Save document (without refresh parameter)
            client = await self.get_client()
            await doc.save(using=client)

            logger.debug(
                "✅ Created episodic memory document successfully: event_id=%s, user_id=%s",
                event_id,
                user_id,
            )
            return doc

        except Exception as e:
            logger.error(
                "❌ Failed to create episodic memory document: event_id=%s, error=%s",
                event_id,
                e,
            )
            raise

    # ==================== Search functionality ====================

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
        explain: bool = False,
        participant_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified search interface using elasticsearch-dsl, supporting multi-word queries and comprehensive filtering

        This method uses elasticsearch-dsl's AsyncSearch class to build queries, providing the same functionality as multi_search,
        but using a more Pythonic DSL syntax instead of directly using the raw client.

        Uses function_score query to implement cumulative scoring based on number of matched terms:
        - Each matched query term increases the document score by 1.0
        - Documents matching more terms are ranked higher
        - Results require at least one term match (min_score=1.0)

        Args:
            query: List of search terms, supports multiple search terms
            user_id: User ID filter
            group_id: Group ID filter
            event_type: Event type filter
            keywords: Keywords filter
            date_range: Time range filter, format: {"gte": "2024-01-01", "lte": "2024-12-31"}
            size: Number of results
            from_: Pagination starting position
            explain: Whether to enable score explanation mode, outputs detailed Elasticsearch scoring process through debug logs

        Returns:
            Hits portion of search results, containing matched document data

        Examples:
            # 1. Multi-word search
            await repo.multi_search(
                query=["company", "Beijing", "technology"],
                user_id="user123",
                size=10
            )

            # 2. Retrieve user memories by time range
            await repo.multi_search(
                query=[],  # Empty query terms
                user_id="user123",
                date_range={"gte": "2024-01-01", "lte": "2024-12-31"},
                size=100
            )

            # 3. Combined query
            await repo.multi_search(
                query=["meeting", "discussion"],
                user_id="user123",
                group_id="group456",
                event_type="Conversation",
                keywords=["work", "project"],
                date_range={"gte": "2024-01-01"}
            )
        """
        try:
            # Create AsyncSearch object
            search = EpisodicMemoryDoc.search()

            # Build filter conditions
            filter_queries = []

            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL:
                if user_id and user_id != "":
                    filter_queries.append(Q("term", user_id=user_id))
                elif user_id is None or user_id == "":
                    # Explicitly filter for null or empty: documents where user_id does not exist
                    filter_queries.append(
                        Q("bool", must_not=Q("exists", field="user_id"))
                    )

            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL:
                if group_id and group_id != "":
                    filter_queries.append(Q("term", group_id=group_id))
                elif group_id is None or group_id == "":
                    # Explicitly filter for null or empty: documents where group_id does not exist
                    filter_queries.append(
                        Q("bool", must_not=Q("exists", field="group_id"))
                    )

            if participant_user_id:
                filter_queries.append(Q("term", participants=participant_user_id))
            if event_type:
                filter_queries.append(Q("term", type=event_type))
            if keywords:
                filter_queries.append(Q("terms", keywords=keywords))
            if date_range:
                filter_queries.append(Q("range", timestamp=date_range))

            # Use different query templates based on whether there are query terms
            if query:
                # ========== Case with query terms: use should clauses in bool query ==========
                #
                # Query structure:
                # bool {
                #   must: [Hard filtering conditions (user_id, group_id, type, keywords, date_range)]
                #   should: [Top 10 query term matching conditions]
                #   minimum_should_match: 1
                # }
                #
                # Scoring rules:
                # 1. Sort query terms by intelligent score, keep top 10 highest scoring terms
                # 2. Each query term in should clause uses boost to set weight (intelligent text score)
                # 3. minimum_should_match=1 ensures at least one term must match to return result
                # 4. Final score = sum of (BM25 score * boost weight) for matched terms

                # Filter query terms by intelligent score, keep top 10 highest scoring terms
                query_with_scores = [
                    (word, self._calculate_text_score(word)) for word in query
                ]
                sorted_query_with_scores = sorted(
                    query_with_scores, key=lambda x: x[1], reverse=True
                )[:10]

                # Build should clauses, each query term uses intelligent text score as boost weight
                should_queries = []
                for word, word_score in sorted_query_with_scores:
                    should_queries.append(
                        Q(
                            "match",
                            search_content={  # Use main field (standard analyzer, will tokenize)
                                "query": word,
                                "boost": word_score,
                            },
                        )
                    )

                # Build bool query
                bool_query_params = {
                    "should": should_queries,
                    "minimum_should_match": 1,  # At least one term must match
                }

                # If there are filter conditions, add to must clause
                if filter_queries:
                    bool_query_params["must"] = filter_queries

                # Use bool query
                search = search.query(Q("bool", **bool_query_params))
            else:
                # ========== Case without query terms: pure filtering query ==========
                #
                # Query structure:
                # bool { filter: [Filter conditions] } or match_all {}
                #
                # Characteristics:
                # 1. No relevance scoring calculated, better performance
                # 2. Sorted by timestamp in descending order
                # 3. Suitable for scenarios like retrieving user memories by time range

                if filter_queries:
                    search = search.query(Q("bool", filter=filter_queries))
                else:
                    search = search.query(Q("match_all"))

                # Sort by timestamp descending when no query terms
                search = search.sort({"timestamp": {"order": "desc"}})

            # Set pagination parameters
            search = search[from_ : from_ + size]

            # Limit returned fields, exclude keywords, linked_entities, extend fields
            # search = search.source(excludes=['keywords', 'linked_entities', 'extend', 'timestamp'])
            # Print search query
            logger.debug("search query: %s", search.to_dict())

            # Execute search
            if explain and query:
                # explain mode: use native client to execute search with explain parameter
                client = await self.get_client()
                index_name = self.get_index_name()

                search_body = search.to_dict()
                search_response = await client.search(
                    index=index_name,
                    body=search_body,
                    explain=True,  # Add explain parameter
                )

                # Convert to standard format and output explanation
                hits = []
                for hit_data in search_response["hits"]["hits"]:
                    # dict_keys(['_shard', '_node', '_index', '_id', '_score', '_source', '_explanation'])
                    hits.append(hit_data)

                    # Output explanation information
                    if "_explanation" in hit_data:
                        explanation = hit_data["_explanation"]
                        self._log_explanation_details(explanation, indent=2)

                logger.debug(
                    "✅ Episodic memory DSL multi-word search succeeded (explain mode): query=%s, user_id=%s, found %d results",
                    search.to_dict(),
                    user_id,
                    len(hits),
                )
            else:
                # Normal mode: use elasticsearch-dsl
                response = await search.execute()

                # Convert to standard format
                hits = []
                for hit in response.hits:
                    hit_data = {
                        "_index": hit.meta.index,
                        "_id": hit.meta.id,
                        "_score": hit.meta.score,
                        "_source": hit.to_dict(),
                    }
                    hits.append(hit_data)

                logger.debug(
                    "✅ Episodic memory DSL multi-word search succeeded: query=%s, user_id=%s, found %d results",
                    search.to_dict(),
                    user_id,
                    len(hits),
                )

            # Return only hits portion, maintain consistent return format with multi_search method
            return hits

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "❌ Episodic memory DSL multi-word search failed: query=%s, user_id=%s, error=%s",
                query,
                user_id,
                e,
            )
            raise
        except Exception as e:
            logger.error(
                "❌ Episodic memory DSL multi-word search failed (unknown error): query=%s, user_id=%s, error=%s",
                query,
                user_id,
                e,
            )
            raise

    async def append_episodic_memory(
        self,
        event_id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        user_name: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        linked_entities: Optional[List[str]] = None,
        subject: Optional[str] = None,
        memcell_event_id_list: Optional[List[str]] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        extend: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> EpisodicMemoryDoc:
        """
        Append episodic memory document

        This is a convenience method that combines document creation and index refresh.
        Suitable for scenarios where newly created documents need to be searchable immediately.

        Args:
            Same as append_episodic_memory method

        Returns:
            Saved EpisodicMemoryDoc instance
        """
        # Create and save document
        doc = await self.create_and_save_episodic_memory(
            event_id=event_id,
            user_id=user_id,
            timestamp=timestamp,
            episode=episode,
            search_content=search_content,
            user_name=user_name,
            title=title,
            summary=summary,
            group_id=group_id,
            participants=participants,
            event_type=event_type,
            keywords=keywords,
            linked_entities=linked_entities,
            subject=subject,
            memcell_event_id_list=memcell_event_id_list,
            parent_type=parent_type,
            parent_id=parent_id,
            extend=extend,
            created_at=created_at,
            updated_at=updated_at,
        )
        return doc

    # ==================== Deletion functionality ====================

    async def delete_by_parent_id(
        self,
        parent_id: str,
        user_id: Optional[str] = None,
        refresh: bool = False,
    ) -> int:
        """
        Delete episodic memory documents by parent MemCell ID.

        Args:
            parent_id: Source MemCell ID
            user_id: If provided, only delete records for this user.
            refresh: Whether to refresh the index immediately

        Returns:
            Number of deleted documents
        """
        try:
            filter_queries: List[Dict[str, Any]] = [{"term": {"parent_id": parent_id}}]
            if user_id is not None:
                filter_queries.append({"term": {"user_id": user_id}})

            delete_query = {"bool": {"must": filter_queries}}
            client = await self.get_client()
            index_name = self.get_index_name()
            response = await client.delete_by_query(
                index=index_name,
                body={"query": delete_query},
                refresh=refresh,
            )
            deleted_count = response.get("deleted", 0)
            logger.debug(
                "✅ Deleted episodic memory by parent_id=%s user_id=%s: %d records",
                parent_id,
                user_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory by parent_id=%s: %s", parent_id, e
            )
            raise

    async def delete_by_event_id(self, event_id: str, refresh: bool = False) -> bool:
        """
        Delete episodic memory document by event_id

        Args:
            event_id: Unique event identifier
            refresh: Whether to refresh index immediately

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            # Use base class delete_by_id method, since we set document ID as event_id
            result = await self.delete_by_id(event_id, refresh=refresh)

            if result:
                logger.debug(
                    "✅ Deleted episodic memory by event_id successfully: event_id=%s",
                    event_id,
                )
            else:
                logger.warning(
                    "⚠️ Failed to delete episodic memory by event_id, document does not exist: event_id=%s",
                    event_id,
                )

            return result

        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory by event_id: event_id=%s, error=%s",
                event_id,
                e,
            )
            raise

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        date_range: Optional[Dict[str, Any]] = None,
        refresh: bool = False,
    ) -> int:
        """
        Batch delete episodic memory documents by filter conditions

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            date_range: Time range filter, format: {"gte": "2024-01-01", "lte": "2024-12-31"}
            refresh: Whether to refresh index immediately

        Returns:
            Number of deleted documents

        Examples:
            # 1. Delete all memories for a specific user
            await repo.delete_by_filters(user_id="user123")

            # 2. Delete memories for a specific user within a specific time range
            await repo.delete_by_filters(
                user_id="user123",
                date_range={"gte": "2024-01-01", "lte": "2024-12-31"}
            )

            # 3. Delete all memories for a specific group
            await repo.delete_by_filters(group_id="group456")

            # 4. Delete with combined conditions
            await repo.delete_by_filters(
                user_id="user123",
                group_id="group456",
                date_range={"gte": "2024-01-01"}
            )
        """
        try:
            # Build filter conditions
            filter_queries = []
            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL and user_id is not None:
                if user_id:  # Non-empty string: personal memories
                    filter_queries.append({"term": {"user_id": user_id}})
                else:  # Empty string: group memories
                    filter_queries.append({"term": {"user_id": ""}})
            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL and group_id:
                filter_queries.append({"term": {"group_id": group_id}})
            if date_range:
                filter_queries.append({"range": {"timestamp": date_range}})

            # At least one filter condition is required to prevent accidental deletion of all data
            if not filter_queries:
                raise ValueError(
                    "At least one filter condition (user_id, group_id or date_range) must be provided"
                )

            # Build delete query
            delete_query = {"bool": {"must": filter_queries}}

            # Execute batch deletion
            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.delete_by_query(
                index=index_name, body={"query": delete_query}, refresh=refresh
            )

            deleted_count = response.get('deleted', 0)

            logger.debug(
                "✅ Batch deleted episodic memory by filter conditions successfully: user_id=%s, group_id=%s, deleted %d records",
                user_id,
                group_id,
                deleted_count,
            )

            return deleted_count

        except ValueError as e:
            logger.error("❌ Deletion parameter error: %s", e)
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to batch delete episodic memory by filter conditions: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            raise

    # ==================== Specialized query methods ====================

    async def get_by_user_and_timerange(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        size: int = 100,
        from_: int = 0,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve memories by user ID and time range

        Args:
            user_id: User ID
            start_time: Start time
            end_time: End time
            size: Number of results
            from_: Pagination starting position

        Returns:
            Search results
        """
        date_range = {}
        if start_time:
            date_range["gte"] = start_time.isoformat()
        if end_time:
            date_range["lte"] = end_time.isoformat()

        return await self.multi_search(
            query=[],  # Empty query terms, pure filtering
            user_id=user_id,
            date_range=date_range if date_range else None,
            size=size,
            from_=from_,
            explain=explain,
        )
