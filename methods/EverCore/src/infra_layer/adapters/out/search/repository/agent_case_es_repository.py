"""
Agent case Elasticsearch repository

Provides BM25 text retrieval for agent case records.
"""

import pprint
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.agent_case import AgentCaseDoc
from core.observation.logger import get_logger
from common_utils.text_utils import SmartTextParser
from core.di.decorators import repository
from biz_layer.retrieve_constants import AGENT_MEMORY_ES_MIN_SHOULD_MATCH

logger = get_logger(__name__)


@repository("agent_case_es_repository", primary=True)
class AgentCaseEsRepository(BaseRepository[AgentCaseDoc]):
    """
    Agent case Elasticsearch repository

    Provides:
    - BM25 text retrieval on task_intents and approaches
    - Multi-term query and filtering capabilities
    - Document creation and management
    """

    def __init__(self):
        super().__init__(AgentCaseDoc)
        self._text_parser = SmartTextParser()

    def _calculate_text_score(self, text: str) -> float:
        """Calculate intelligent score of text for boost weighting."""
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
        pprint.pprint(explanation, indent=indent)

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified search interface for agent case BM25 retrieval

        Args:
            query: List of search terms
            user_id: User ID filter
            session_id: Session ID filter
            group_ids: List of Group IDs to filter
            parent_id: Parent memory ID filter
            date_range: Time range filter
            size: Number of results
            from_: Pagination start position
            explain: Whether to enable score explanation mode

        Returns:
            Hits part of search results
        """
        try:
            search = AgentCaseDoc.search()

            filter_queries = []

            if user_id != MAGIC_ALL:
                if user_id and user_id != "":
                    filter_queries.append(Q("term", user_id=user_id))
                else:
                    # user_id must not exist: match docs where field is missing or ""
                    filter_queries.append(
                        Q(
                            "bool",
                            should=[
                                Q("bool", must_not=[Q("exists", field="user_id")]),
                                Q("term", user_id=""),
                            ],
                            minimum_should_match=1,
                        )
                    )

            if session_id:
                filter_queries.append(Q("term", session_id=session_id))

            if group_ids is not None and len(group_ids) > 0:
                filter_queries.append(Q("terms", group_id=group_ids))

            if parent_id:
                filter_queries.append(Q("term", parent_id=parent_id))

            if date_range:
                filter_queries.append(Q("range", timestamp=date_range))

            if query:
                query_with_scores = [
                    (word, self._calculate_text_score(word)) for word in query
                ]
                sorted_query_with_scores = sorted(
                    query_with_scores, key=lambda x: x[1], reverse=True
                )[:10]

                should_queries = []
                for word, word_score in sorted_query_with_scores:
                    should_queries.append(
                        Q("match", search_content={"query": word, "boost": word_score})
                    )

                bool_query_params = {
                    "should": should_queries,
                    "minimum_should_match": AGENT_MEMORY_ES_MIN_SHOULD_MATCH,
                }

                if filter_queries:
                    bool_query_params["must"] = filter_queries

                search = search.query(Q("bool", **bool_query_params))
            else:
                if filter_queries:
                    search = search.query(Q("bool", filter=filter_queries))
                else:
                    search = search.query(Q("match_all"))

                search = search.sort({"timestamp": {"order": "desc"}})

            search = search[from_ : from_ + size]

            logger.debug("agent case search query: %s", search.to_dict())

            if explain and query:
                client = await self.get_client()
                index_name = self.get_index_name()

                search_body = search.to_dict()
                search_response = await client.search(
                    index=index_name, body=search_body, explain=True
                )

                hits = []
                for hit_data in search_response["hits"]["hits"]:
                    hits.append(hit_data)
                    if "_explanation" in hit_data:
                        self._log_explanation_details(
                            hit_data["_explanation"], indent=2
                        )

                logger.debug(
                    "Agent case search succeeded (explain mode): query=%s, found %d results",
                    search.to_dict(),
                    len(hits),
                )
            else:
                response = await search.execute()

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
                    "Agent case search succeeded: query=%s, found %d results",
                    search.to_dict(),
                    len(hits),
                )

            return hits

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Agent case search failed: query=%s, error=%s", query, e)
            raise
        except Exception as e:
            logger.error(
                "Agent case search failed (unknown error): query=%s, error=%s", query, e
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
        Batch delete agent case documents by filter conditions

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
            if user_id != MAGIC_ALL and user_id is not None:
                if user_id:
                    filter_queries.append({"term": {"user_id": user_id}})
                else:
                    filter_queries.append({"term": {"user_id": ""}})
            if group_id != MAGIC_ALL and group_id:
                filter_queries.append({"term": {"group_id": group_id}})
            if date_range:
                filter_queries.append({"range": {"timestamp": date_range}})

            if not filter_queries:
                raise ValueError("At least one filter condition must be provided")

            delete_query = {"bool": {"must": filter_queries}}

            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.delete_by_query(
                index=index_name, body={"query": delete_query}, refresh=refresh
            )

            deleted_count = response.get("deleted", 0)
            logger.info(
                "Successfully deleted agent case docs: user_id=%s, group_id=%s, deleted %d records",
                user_id,
                group_id,
                deleted_count,
            )
            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to delete agent case docs: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            raise
