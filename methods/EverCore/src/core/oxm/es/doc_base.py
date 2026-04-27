import typing
from typing import Type, Any, Dict, Set

import os
from fnmatch import fnmatch
from datetime import datetime
from elasticsearch.dsl import MetaField, AsyncDocument, field as e_field
from common_utils.datetime_utils import to_timezone
from core.oxm.es.es_utils import generate_index_name, get_index_ns
from core.oxm.es.mapping_templates import DYNAMIC_TEMPLATES
from elasticsearch import AsyncElasticsearch


class DocBase(AsyncDocument):
    """Elasticsearch document base class"""

    @classmethod
    def get_connection(cls) -> AsyncElasticsearch:
        """
        Get connection
        """
        return cls._get_connection()

    @classmethod
    def get_index_name(cls) -> str:
        """
        Get index name (alias)

        Returns:
            str: Index alias

        Raises:
            ValueError: If the document class does not have correct index configuration
        """
        if hasattr(cls, '_index') and hasattr(cls._index, '_name'):
            return cls._index._name
        raise ValueError(
            f"Document class {cls.__name__} does not have correct index configuration"
        )


class AliasSupportDoc(DocBase):
    """Document class supporting alias pattern, enhanced with timezone handling for date fields"""

    class CustomMeta:
        # Specify the field name used to automatically populate meta.id (e.g., MongoDB primary key field), not enabled if not set
        id_source_field: typing.Optional[str] = None
        # Cache set of Date-type field names for quick checking (dynamically set, no need to predefine)
        # date_fields: typing.Optional[Set[str]] = None

    @classmethod
    def _init_date_fields_cache(cls) -> Set[str]:
        """
        Initialize Date field cache, collect all Date-type field names in the class

        Returns:
            Set of Date-type field names
        """
        # Get cached date_fields from CustomMeta
        custom_meta = getattr(cls, 'CustomMeta', None)
        if custom_meta is not None:
            existing_cache = getattr(custom_meta, 'date_fields', None)
            if existing_cache is not None:
                return existing_cache

        date_fields = set()
        # Iterate through all class attributes to find Date-type fields
        for attr_name in dir(cls):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, e_field.Date):
                    date_fields.add(attr_name)
            except (AttributeError, TypeError):
                # Ignore attributes that cannot be accessed or are not fields
                continue

        # Dynamically set to CustomMeta
        if custom_meta is not None:
            setattr(custom_meta, 'date_fields', date_fields)

        return date_fields

    def _process_date_field(self, field_name: str, field_value: Any) -> Any:
        """
        Process date field to ensure timezone information is present

        Args:
            field_name: Field name
            field_value: Field value

        Returns:
            Processed field value
        """
        # Use cached Date field set for quick checking
        # Ensure non-None Set[str] by calling _init_date_fields_cache
        date_fields = self.__class__._init_date_fields_cache()
        if field_name in date_fields and isinstance(field_value, datetime):
            return to_timezone(field_value)
        return field_value

    def __setattr__(self, name: str, value: Any) -> None:
        """Override field setting method to apply timezone processing for date fields"""
        # Apply timezone processing for date fields
        processed_value = self._process_date_field(name, value)

        # Call parent class __setattr__
        super().__setattr__(name, processed_value)

    def __init__(self, meta: Dict[str, Any] = None, **kwargs: Any):
        """Override constructor: set meta.id strictly based on explicit ID_SOURCE_FIELD, raise error if missing"""

        # Initialize Date field cache (will initialize on first call, then use cache)
        self.__class__._init_date_fields_cache()

        raw_kwargs = dict(kwargs)

        # Process date fields in kwargs
        processed_kwargs = {}
        for field_name, field_value in raw_kwargs.items():
            processed_kwargs[field_name] = self._process_date_field(
                field_name, field_value
            )

        # Strictly set meta.id based on ID_SOURCE_FIELD (no heuristics), and compatible with ES construction (meta with _id)
        # Get id_source_field configuration from CustomMeta
        custom_meta_class = getattr(self.__class__, 'CustomMeta', None)

        id_source_field = (
            getattr(custom_meta_class, 'id_source_field', None)
            if custom_meta_class
            else None
        )
        merged_meta: Dict[str, Any] = {} if meta is None else dict(meta)
        # Extract provided meta id (from ES loading scenario)
        given_meta_id = None
        if "id" in merged_meta and merged_meta["id"] not in (None, ""):
            given_meta_id = merged_meta["id"]
        if "_id" in merged_meta and merged_meta["_id"] not in (None, ""):
            if given_meta_id is not None and given_meta_id != merged_meta["_id"]:
                raise ValueError("meta.id conflicts between 'id' and '_id'")
            given_meta_id = merged_meta["_id"]

        if given_meta_id is not None:
            # If meta id is explicitly provided, validate consistency with ID_SOURCE_FIELD (if exists)
            if id_source_field and id_source_field in processed_kwargs:
                source_value = processed_kwargs[id_source_field]
                if source_value not in (None, "") and source_value != given_meta_id:
                    raise ValueError(
                        "meta.id conflicts with value from ID_SOURCE_FIELD"
                    )
            # Normalize meta fields
            merged_meta["id"] = given_meta_id
            merged_meta["_id"] = given_meta_id
        elif id_source_field:
            # If meta id is not provided, require it from ID_SOURCE_FIELD
            if id_source_field not in processed_kwargs or processed_kwargs[
                id_source_field
            ] in (None, ""):
                raise ValueError(
                    f"{self.__class__.__name__} requires non-empty '{id_source_field}' to set meta.id"
                )
            source_value = processed_kwargs[id_source_field]
            merged_meta["id"] = source_value
            merged_meta["_id"] = source_value

        # Call parent constructor
        super().__init__(merged_meta or None, **processed_kwargs)

    @classmethod
    def _matches(cls, hit):
        # override _matches to match indices in a pattern instead of just ALIAS
        # hit is the raw dict as returned by elasticsearch
        return fnmatch(hit["_index"], cls.PATTERN)

    @classmethod
    def dest(cls):
        return generate_index_name(cls)


def AliasDoc(
    doc_name: str,
    number_of_shards: int = 3,
    number_of_replicas: int = 1,
    refresh_interval: str = "10s",
) -> Type[AsyncDocument]:
    """
    Create an ES document class supporting alias pattern

    Automatically handle timezone for date fields:
    - For int timestamps, no processing
    - For datetime objects without timezone, automatically add current system timezone
    - Ensure all date fields have timezone information to avoid timezone-related issues

    Args:
        doc_name: Document name
        build_analyzers: Optional list of analyzers
        number_of_shards: Number of shards

    Returns:
        Enhanced document class
    """

    if get_index_ns():
        doc_name = f"{doc_name}-{get_index_ns()}"

    class GeneratedAliasSupportDoc(AliasSupportDoc):
        PATTERN = f"{doc_name}-*"

        class Index:
            name = doc_name
            settings = {
                "number_of_shards": number_of_shards,
                "number_of_replicas": number_of_replicas,
                "refresh_interval": refresh_interval,
                "max_ngram_diff": 50,
                "max_shingle_diff": 10,
            }

        class Meta:
            dynamic = MetaField("true")
            # Disable date auto-detection to prevent "2023/10/01" from being incorrectly converted and causing subsequent errors
            date_detection = MetaField(False)
            # Disable numeric detection to prevent string numbers from being confused
            numeric_detection = MetaField(False)
            # Dynamic mapping rules based on field suffixes (see mapping_templates.py)
            dynamic_templates = MetaField(DYNAMIC_TEMPLATES)

    return GeneratedAliasSupportDoc
