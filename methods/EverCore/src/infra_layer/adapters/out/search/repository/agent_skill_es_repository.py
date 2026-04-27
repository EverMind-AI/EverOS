"""
Agent skill Elasticsearch repository

Provides BM25 text retrieval for agent skill records.
"""

import pprint
from typing import List, Optional, Dict, Any
from elasticsearch.dsl import Q
from core.oxm.es.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.elasticsearch.memory.agent_skill import (
    AgentSkillDoc,
)
from core.observation.logger import get_logger
from biz_layer.retrieve_constants import AGENT_MEMORY_ES_MIN_SHOULD_MATCH
from common_utils.text_utils import SmartTextParser
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("agent_skill_es_repository", primary=True)
class AgentSkillEsRepository(BaseRepository[AgentSkillDoc]):
    """
    Agent skill Elasticsearch repository

    Provides:
    - BM25 text retrieval on skill name + description
    - Multi-term query and filtering capabilities
    - Document creation and management
    """

    def __init__(self):
        super().__init__(AgentSkillDoc)
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
        group_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        size: int = 10,
        from_: int = 0,
        explain: bool = False,
        maturity_threshold: Optional[float] = 0.6,
        confidence_threshold: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified search interface for agent skill BM25 retrieval

        Args:
            query: List of search terms
            group_ids: List of Group IDs to filter
            user_id: User ID filter
            cluster_id: Cluster ID filter
            size: Number of results
            from_: Pagination start position
            explain: Whether to enable score explanation mode
            maturity_threshold: Minimum maturity score (0.0-1.0) to return.
                Set to None to include all skills regardless of maturity.
            confidence_threshold: Minimum confidence score (0.0-1.0) to return.
                Skills below this threshold are considered retired.
                Set to None to skip confidence filtering.

        Returns:
            Hits part of search results
        """
        try:
            search = AgentSkillDoc.search()

            filter_queries = []

            if maturity_threshold is not None:
                filter_queries.append(
                    Q("range", maturity_score={"gte": maturity_threshold})
                )

            if confidence_threshold is not None:
                filter_queries.append(
                    Q("range", confidence={"gte": confidence_threshold})
                )

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

            if group_ids is not None and len(group_ids) > 0:
                filter_queries.append(Q("terms", group_id=group_ids))

            if cluster_id:
                filter_queries.append(Q("term", cluster_id=cluster_id))

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

                search = search.sort({"maturity_score": {"order": "desc"}})

            search = search[from_ : from_ + size]

            logger.debug("agent skill search query: %s", search.to_dict())

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
                    "Agent skill search succeeded (explain mode): query=%s, found %d results",
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
                    "Agent skill search succeeded: query=%s, found %d results",
                    search.to_dict(),
                    len(hits),
                )

            return hits

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Agent skill search failed: query=%s, error=%s", query, e)
            raise
        except Exception as e:
            logger.error(
                "Agent skill search failed (unknown error): query=%s, error=%s",
                query,
                e,
            )
            raise

    async def delete_by_filters(
        self,
        group_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        refresh: bool = False,
    ) -> int:
        """
        Batch delete agent skill documents by filter conditions

        Args:
            group_id: Group ID filter
            cluster_id: Cluster ID filter
            refresh: Whether to refresh index immediately

        Returns:
            Number of deleted documents
        """
        try:
            filter_queries = []
            if group_id:
                filter_queries.append({"term": {"group_id": group_id}})
            if cluster_id:
                filter_queries.append({"term": {"cluster_id": cluster_id}})

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
                "Successfully deleted agent skill docs: cluster_id=%s, deleted %d records",
                cluster_id,
                deleted_count,
            )
            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to delete agent skill docs: cluster_id=%s, error=%s",
                cluster_id,
                e,
            )
            raise
