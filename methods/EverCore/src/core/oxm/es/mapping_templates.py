"""
Elasticsearch Dynamic Mapping Templates

Dynamic mapping rules configuration based on field suffixes, used to centrally manage field type inference for ES indexes.

Naming conventions:
- *_ts, *_date  → date      (date and time)
- *_num         → long      (integer)
- *_desc        → text      (full-text search, tokenized)
- *_flag        → boolean   (boolean flag)
- *_enabled     → boolean   (enabled status)
- *_rate        → double    (floating-point number/ratio)
- *_id          → keyword   (ID, exact match)
- other strings → keyword   (fallback, exact match)
"""

# Dynamic mapping rules based on field suffixes (matched in order, first match takes effect)
DYNAMIC_TEMPLATES = [
    # 1. Date rule: match *_ts suffix
    {
        "rule_date_ts": {
            "match": "*_ts",
            "match_mapping_type": "*",
            "mapping": {"type": "date"},
        }
    },
    # 2. Date rule: match *_date suffix
    {
        "rule_date": {
            "match": "*_date",
            "match_mapping_type": "*",
            "mapping": {"type": "date"},
        }
    },
    # 3. Numeric rule: match *_num suffix, force to long
    {
        "rule_long": {
            "match": "*_num",
            "match_mapping_type": "long",
            "mapping": {"type": "long"},
        }
    },
    # 4. Full-text search rule: match *_desc suffix, requires tokenization
    {
        "rule_text": {
            "match": "*_desc",
            "match_mapping_type": "string",
            "mapping": {"type": "text", "analyzer": "standard"},
        }
    },
    # 5. Boolean rule: match *_flag suffix
    {
        "rule_bool_flag": {
            "match": "*_flag",
            "match_mapping_type": "boolean",
            "mapping": {"type": "boolean"},
        }
    },
    # 6. Boolean rule: match *_enabled suffix
    {
        "rule_bool_enabled": {
            "match": "*_enabled",
            "match_mapping_type": "boolean",
            "mapping": {"type": "boolean"},
        }
    },
    # 7. Floating-point rule: match *_rate suffix
    {
        "rule_double": {
            "match": "*_rate",
            "match_mapping_type": "double",
            "mapping": {"type": "double"},
        }
    },
    # 8. ID rule: match *_id suffix, exact match without tokenization
    {
        "rule_id": {
            "match": "*_id",
            "match_mapping_type": "string",
            "mapping": {"type": "keyword"},
        }
    },
    # 9. Fallback rule: all other strings → keyword
    {
        "strings_as_keywords": {
            "match_mapping_type": "string",
            "mapping": {"type": "keyword", "ignore_above": 256},
        }
    },
]
