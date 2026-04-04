"""
Foresight Elasticsearch Repository

Foresight-specific repository class based on BaseRepository, providing efficient BM25 text retrieval and complex query capabilities.
Reuses EpisodicMemoryDoc, filtering by type field as foresight.
"""

from datetime import datetime
import pprint
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.foresight import ForesightDoc
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from common_utils.text_utils import SmartTextParser
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("foresight_es_repository", primary=True)
class ForesightEsRepository(BaseRepository[ForesightDoc]):
    """
    Foresight Elasticsearch Repository

    Specialized repository class based on BaseRepository, providing:
    - Efficient BM25 text retrieval
    - Multi-term query and filtering capabilities
    - Document creation and management
    - Manual index refresh control

    Note: Reuses EpisodicMemoryDoc, filtering by type field as foresight.
    """

    def __init__(self):
        """Initialize foresight repository"""
        super().__init__(ForesightDoc)
        # Initialize smart text parser for calculating intelligent length of query terms
        self._text_parser = SmartTextParser()

    def _calculate_text_score(self, text: str) -> float:
        """
        Calculate intelligent score of text

        Uses SmartTextParser to compute total score of text, considering weights of different types such as CJK characters, English words, etc.

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

    async def create_and_save_foresight(
        self,
        id: str,
        user_id: str,
        user_name: str,
        timestamp: datetime,
        content: str,
        search_content: List[str],
        parent_id: str,
        parent_type: str,
        event_type: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        extend: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> ForesightDoc:
        """
        Create and save foresight document

        Args:
            id: Unique identifier for memory
            user_id: User ID (required)
            timestamp: Event occurrence time (required)
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
            extend: Extension fields
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

            # Build extend field, including foresight-specific information
            foresight_extend = extend or {}
            foresight_extend.update(
                {
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "duration_days": duration_days,
                }
            )

            # Create document instance
            doc = ForesightDoc(
                id=id,
                type=event_type,
                user_id=user_id,
                user_name=user_name or '',
                timestamp=timestamp,
                foresight=content,
                search_content=search_content,
                evidence=evidence or '',
                group_id=group_id,
                participants=participants or [],
                parent_type=parent_type,
                parent_id=parent_id,
                extend=foresight_extend,
                created_at=created_at,
                updated_at=updated_at,
            )

            # Save document (without refresh parameter)
            client = await self.get_client()
            await doc.save(using=client)

            logger.debug(
                "✅ Created foresight document successfully: id=%s, user_id=%s",
                id,
                user_id,
            )
            return doc

        except Exception as e:
            logger.error(
                "❌ Failed to create foresight document: id=%s, error=%s", id, e
            )
            raise

    # ==================== Search functionality ====================

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
        explain: bool = False,
        participant_user_id: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Unified search interface using elasticsearch-dsl, supporting multi-term queries and comprehensive filtering

        Uses function_score query to achieve cumulative scoring based on number of matching terms.
        Automatically filters documents with type="foresight".

        Args:
            query: List of search terms, supports multiple terms
            user_id: User ID filter
            group_id: Group ID filter
            parent_type: Parent type filter (e.g., "memcell", "episode")
            parent_id: Parent memory ID filter
            keywords: Keyword filter
            date_range: Time range filter, format: {"gte": "2024-01-01", "lte": "2024-12-31"}
            size: Number of results
            from_: Pagination starting position
            explain: Whether to enable score explanation mode
            participant_user_id: When retrieving group data, additionally require this user to be a participant
            current_time: Current time (only used when filtering by start/end validity period)

        Returns:
            Hits portion of search results, containing matched document data
        """
        try:
            # Create AsyncSearch object
            search = ForesightDoc.search()

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

            # Handle parent_id filter
            if parent_id:
                filter_queries.append(Q("term", parent_id=parent_id))

            # Handle parent_type filter
            if parent_type:
                filter_queries.append(Q("term", parent_type=parent_type))

            if keywords:
                filter_queries.append(Q("terms", keywords=keywords))
            if date_range:
                filter_queries.append(Q("range", timestamp=date_range))

            # Use different query templates based on whether there are query terms
            if query:
                # Filter query terms by intelligent score, keep top 10 highest scoring terms
                query_with_scores = [
                    (word, self._calculate_text_score(word)) for word in query
                ]
                sorted_query_with_scores = sorted(
                    query_with_scores, key=lambda x: x[1], reverse=True
                )[:10]

                # Build should clauses
                should_queries = []
                for word, word_score in sorted_query_with_scores:
                    should_queries.append(
                        Q("match", search_content={"query": word, "boost": word_score})
                    )

                # Build bool query parameters
                bool_query_params = {
                    "should": should_queries,
                    "minimum_should_match": 1,
                }

                # If there are filter conditions, add to must clause
                if filter_queries:
                    bool_query_params["must"] = filter_queries

                # Use bool query
                search = search.query(Q("bool", **bool_query_params))
            else:
                # Case without query terms: pure filtering query
                if filter_queries:
                    search = search.query(Q("bool", filter=filter_queries))
                else:
                    search = search.query(Q("match_all"))

                # Sort by timestamp descending when no query terms
                search = search.sort({"timestamp": {"order": "desc"}})

            # Set pagination parameters
            search = search[from_ : from_ + size]

            logger.debug("foresight search query: %s", search.to_dict())

            # Execute search
            if explain and query:
                # explain mode
                client = await self.get_client()
                index_name = self.get_index_name()

                search_body = search.to_dict()
                search_response = await client.search(
                    index=index_name, body=search_body, explain=True
                )

                # Convert to standard format and output explanation
                hits = []
                for hit_data in search_response["hits"]["hits"]:
                    hits.append(hit_data)

                    # Output explanation information
                    if "_explanation" in hit_data:
                        explanation = hit_data["_explanation"]
                        self._log_explanation_details(explanation, indent=2)

                logger.debug(
                    "✅ Foresight DSL multi-term search succeeded (explain mode): query=%s, user_id=%s, found %d results",
                    search.to_dict(),
                    user_id,
                    len(hits),
                )
            else:
                # Normal mode
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

            # Filter by validity period based on current_time
            if current_time:
                current_dt = current_time
                if isinstance(current_dt, str):
                    try:
                        current_dt = datetime.fromisoformat(current_dt)
                    except ValueError:
                        try:
                            current_dt = datetime.fromisoformat(
                                current_dt.replace("Z", "+00:00")
                            )
                        except ValueError:
                            current_dt = None
                filtered_hits = []
                for hit_data in hits:
                    source = hit_data.get("_source", {}) or {}
                    extend = source.get("extend") or {}
                    start_dt = self._parse_datetime(extend.get("start_time"))
                    end_dt = self._parse_datetime(extend.get("end_time"))
                    if start_dt and current_dt and start_dt > current_dt:
                        continue
                    if end_dt and current_dt and end_dt < current_dt:
                        continue
                    filtered_hits.append(hit_data)
                hits = filtered_hits

                logger.debug(
                    "✅ Foresight DSL multi-term search succeeded: query=%s, user_id=%s, found %d results",
                    search.to_dict(),
                    user_id,
                    len(hits),
                )

            # Return only hits portion
            return hits

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "❌ Foresight DSL multi-term search failed: query=%s, user_id=%s, error=%s",
                query,
                user_id,
                e,
            )
            raise
        except Exception as e:
            logger.error(
                "❌ Foresight DSL multi-term search failed (unknown error): query=%s, user_id=%s, error=%s",
                query,
                user_id,
                e,
            )
            raise

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string"""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

    # ==================== Deletion functionality ====================

    async def delete_expired(
        self,
        before: datetime,
        refresh: bool = False,
    ) -> int:
        """
        Delete foresight documents whose validity period has ended before the given time.

        The end_time is stored as an ISO string inside the ``extend`` field.
        We use a range query on ``extend.end_time`` to find expired records.

        Args:
            before: Delete foresights with end_time strictly before this datetime.
            refresh: Whether to refresh the index immediately

        Returns:
            Number of deleted documents
        """
        try:
            # extend.end_time is stored as ISO string, e.g. "2024-03-01T00:00:00"
            before_str = before.isoformat()
            delete_query = {
                "range": {
                    "extend.end_time": {"lt": before_str}
                }
            }
            client = await self.get_client()
            index_name = self.get_index_name()
            response = await client.delete_by_query(
                index=index_name,
                body={"query": delete_query},
                refresh=refresh,
            )
            deleted_count = response.get("deleted", 0)
            logger.info(
                "✅ Deleted expired foresights from ES (before=%s): %d records",
                before_str,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("❌ Failed to delete expired foresights from ES: %s", e)
            raise
