"""
Foresight Elasticsearch Repository

V1 simplified repository for BM25 text retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using parent_id.
"""

import pprint
from datetime import datetime
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.foresight import ForesightDoc
from core.observation.logger import get_logger
from common_utils.text_utils import SmartTextParser
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("foresight_es_repository", primary=True)
class ForesightEsRepository(BaseRepository[ForesightDoc]):
    """
    Foresight Elasticsearch Repository

    V1 simplified repository for BM25 text retrieval.
    Only stores search-essential fields in ES.
    Full data is retrieved from MongoDB using parent_id.
    """

    def __init__(self):
        """Initialize foresight repository"""
        super().__init__(ForesightDoc)
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

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """
        Parse a datetime value from various formats

        Args:
            value: Value to parse (string, datetime, or None)

        Returns:
            Parsed datetime or None
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
        return None

    # ==================== Document creation and management ====================

    async def create_and_save_foresight(
        self,
        id: str,
        user_id: str,
        content: str,
        search_content: List[str],
        parent_id: str,
        parent_type: str,
        event_type: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> ForesightDoc:
        """
        Create and save foresight document

        Args:
            id: Unique identifier for memory
            user_id: User ID (required)
            content: Foresight content (required)
            search_content: List of search content (supports multiple search terms, required)
            parent_id: Parent memory ID
            parent_type: Parent memory type (memcell/episode)
            group_id: Group ID
            participants: List of participants
            start_time: Validity start time
            end_time: Validity end time
            duration_days: Duration in days
            evidence: Evidence (original factual basis)
            created_at: Creation time
            updated_at: Update time

        Returns:
            Saved ForesightDoc instance
        """
        try:
            # Set default timestamp
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # Create document instance
            doc = ForesightDoc(
                id=id,
                type=event_type,
                user_id=user_id,
                foresight=content,
                search_content=search_content,
                evidence=evidence or '',
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
                "Created foresight document successfully: id=%s, user_id=%s",
                id,
                user_id,
            )
            return doc

        except Exception as e:
            logger.error("Failed to create foresight document: id=%s, error=%s", id, e)
            raise

    # ==================== Search functionality ====================

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        sender_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        BM25 text search for foresight records

        Args:
            query: List of search terms
            user_id: User ID filter
            group_ids: List of Group IDs to filter
            sender_id: Sender ID filter
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
            search = ForesightDoc.search()

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

            # Handle sender_id filter (match against sender_ids array)
            if sender_id:
                filter_queries.append(Q("term", sender_ids=sender_id))

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
                filter_queries.append(Q("range", created_at=date_range))

            # Build query
            if query:
                # Filter query terms by intelligent score
                query_with_scores = [
                    (word, self._calculate_text_score(word)) for word in query
                ]
                sorted_query_with_scores = sorted(
                    query_with_scores, key=lambda x: x[1], reverse=True
                )[:10]

                # Build should clauses - no text fields in ForesightDoc for BM25
                # Only filtering is supported, rely on Milvus for vector search
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

                search = search.sort({"created_at": {"order": "desc"}})

            search = search[from_ : from_ + size]

            logger.debug("foresight search query: %s", search.to_dict())

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
                "Foresight search succeeded: query=%s, user_id=%s, found %d results",
                query,
                user_id,
                len(hits),
            )

            return hits

        except Exception as e:
            logger.error(
                "Foresight search failed: query=%s, user_id=%s, error=%s",
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
        Batch delete foresight documents by filter conditions

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
                filter_queries.append({"range": {"created_at": date_range}})

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
                "Batch deleted foresights: user_id=%s, group_id=%s, deleted %d records",
                user_id,
                group_id,
                deleted_count,
            )
            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to batch delete foresights: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            raise
