"""
Atomic Fact Elasticsearch Repository

V1 simplified repository for BM25 text retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using parent_id.
"""

import pprint
from datetime import datetime
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.atomic_fact import (
    AtomicFactDoc,
)
from core.observation.logger import get_logger
from common_utils.text_utils import SmartTextParser
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("atomic_fact_es_repository", primary=True)
class AtomicFactEsRepository(BaseRepository[AtomicFactDoc]):
    """
    Atomic Fact Elasticsearch Repository

    V1 simplified repository for BM25 text retrieval.
    Only stores search-essential fields in ES.
    Full data is retrieved from MongoDB using parent_id.
    """

    def __init__(self):
        """Initialize atomic fact repository"""
        super().__init__(AtomicFactDoc)
        self._text_parser = SmartTextParser()

    def _calculate_text_score(self, text: str) -> float:
        """Calculate intelligent score of text"""
        if not text:
            return 0.0
        try:
            tokens = self._text_parser.parse_tokens(text)
            return self._text_parser.calculate_total_score(tokens)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "Failed to calculate text score, using character length: %s", e
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

    async def create_and_save_atomic_fact(
        self,
        id: str,
        user_id: str,
        timestamp: datetime,
        atomic_fact: str,
        search_content: List[str],
        parent_id: str,
        parent_type: str,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> AtomicFactDoc:
        """
        Create and save atomic fact document

        Args:
            id: Log unique identifier
            user_id: User ID (required)
            timestamp: Event occurrence time (required)
            atomic_fact: Atomic fact (required)
            search_content: List of search content (supports multiple search terms, required)
            parent_id: Parent memory ID
            parent_type: Parent memory type (memcell/episode)
            group_id: Group ID
            participants: List of participants
            created_at: Creation time
            updated_at: Update time

        Returns:
            Saved AtomicFactDoc instance
        """
        try:
            # Set default timestamps
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # Create document instance
            doc = AtomicFactDoc(
                id=id,
                user_id=user_id,
                timestamp=timestamp,
                search_content=search_content,
                atomic_fact=atomic_fact,
                group_id=group_id,
                participants=participants or [],
                sender_ids=sender_ids or [],
                parent_type=parent_type,
                parent_id=parent_id,
                created_at=created_at,
                updated_at=updated_at,
            )

            # Save document (without refresh parameter)
            client = await self.get_client()
            await doc.save(using=client)

            logger.debug(
                "Created atomic fact document successfully: event_id=%s, user_id=%s",
                id,
                user_id,
            )
            return doc

        except Exception as e:
            logger.error(
                "Failed to create atomic fact document: event_id=%s, error=%s", id, e
            )
            raise

    # ==================== Search functionality ====================

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        BM25 text search for atomic facts

        Args:
            query: List of search terms
            user_id: User ID filter
            group_ids: List of Group IDs to filter
            session_id: Session ID filter
            parent_type: Parent type filter
            parent_id: Parent memory ID filter
            date_range: Time range filter
            size: Number of results
            from_: Pagination start position

        Returns:
            List of search hits with matched document data
        """
        try:
            search = AtomicFactDoc.search()

            # Build filter conditions
            filter_queries = []

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id and user_id != "":
                    filter_queries.append(Q("term", user_id=user_id))
                else:
                    # user_id must not exist: match docs where field is missing or ""
                    filter_queries.append(
                        Q("bool", should=[
                            Q("bool", must_not=[Q("exists", field="user_id")]),
                            Q("term", user_id=""),
                        ], minimum_should_match=1)
                    )

            # Handle group_ids filter
            if group_ids is not None and len(group_ids) > 0:
                filter_queries.append(Q("terms", group_id=group_ids))

            # Handle session_id filter
            if session_id:
                filter_queries.append(Q("term", session_id=session_id))

            # Handle parent_id filter
            if parent_id:
                filter_queries.append(Q("term", parent_id=parent_id))

            # Handle parent_type filter
            if parent_type:
                filter_queries.append(Q("term", parent_type=parent_type))

            if date_range:
                filter_queries.append(Q("range", timestamp=date_range))

            # Build query
            if query:
                # Filter query terms by intelligent score
                query_with_scores = [
                    (word, self._calculate_text_score(word)) for word in query
                ]
                sorted_query_with_scores = sorted(
                    query_with_scores, key=lambda x: x[1], reverse=True
                )[:10]

                # Build should clauses - search in atomic_fact field
                should_queries = []
                for word, word_score in sorted_query_with_scores:
                    should_queries.append(
                        Q("match", search_content={"query": word, "boost": word_score})
                    )

                bool_query_params = {
                    "should": should_queries,
                    "minimum_should_match": 1,
                }

                if filter_queries:
                    bool_query_params["must"] = filter_queries

                search = search.query(Q("bool", **bool_query_params))
            else:
                # Pure filtering query
                if filter_queries:
                    search = search.query(Q("bool", filter=filter_queries))
                else:
                    search = search.query(Q("match_all"))

                search = search.sort({"timestamp": {"order": "desc"}})

            search = search[from_ : from_ + size]

            logger.debug("atomic fact search query: %s", search.to_dict())

            response = await search.execute()

            hits = []
            for hit in response.hits:
                hit_data = {
                    "_id": hit.meta.id,
                    "_score": hit.meta.score,
                    "_source": hit.to_dict(),
                }
                hits.append(hit_data)

            logger.debug(
                "Atomic fact search succeeded: query=%s, user_id=%s, found %d results",
                query,
                user_id,
                len(hits),
            )

            return hits

        except Exception as e:
            logger.error(
                "Atomic fact search failed: query=%s, user_id=%s, error=%s",
                query,
                user_id,
                e,
            )
            raise

    # ==================== Deletion functionality ====================

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        date_range: Optional[Dict[str, Any]] = None,
        refresh: bool = False,
    ) -> int:
        """
        Batch delete atomic fact documents by filter conditions

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            date_range: Time range filter
            refresh: Whether to refresh index immediately

        Returns:
            Number of deleted documents
        """
        try:
            filter_queries = []
            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL:
                if not user_id:  # None or "" -> match empty string
                    filter_queries.append({"term": {"user_id": ""}})
                else:
                    filter_queries.append({"term": {"user_id": user_id}})
            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL:
                if not group_id:  # None or "" -> match empty string
                    filter_queries.append({"term": {"group_id": ""}})
                else:
                    filter_queries.append({"term": {"group_id": group_id}})
            if date_range:
                filter_queries.append({"range": {"timestamp": date_range}})

            if not filter_queries:
                raise ValueError(
                    "At least one filter condition (user_id, group_id or date_range) must be provided"
                )

            delete_query = {"bool": {"must": filter_queries}}

            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.delete_by_query(
                index=index_name, body={"query": delete_query}, refresh=refresh
            )

            deleted_count = response.get('deleted', 0)
            logger.info(
                "Batch deleted atomic facts: user_id=%s, group_id=%s, deleted %d records",
                user_id,
                group_id,
                deleted_count,
            )
            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to batch delete atomic facts: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            raise
