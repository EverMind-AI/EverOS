from elasticsearch.dsl import tokenizer, normalizer, token_filter, analyzer

# ================================
# Token Filters definition
# ================================

# Shingle filter - used to generate consecutive phrases (with space separation)
# For example: "hello world test" -> ["hello world", "world test", "hello world test"]
# Does not output single words (output_unigrams=False), only outputs phrases
shingle_lease_filter = token_filter(
    "shingle_lease",
    type="shingle",
    min_shingle_size=2,  # Minimum phrase length: 2 words
    max_shingle_size=5,  # Maximum phrase length: 5 words
    output_unigrams=False,  # Do not output single words
)

# Shingle filter - used to generate consecutive phrases (without space separation)
# For example: "hello world test" -> ["helloworld", "worldtest", "helloworldtest"]
# Suitable for Chinese or tightly connected phrases
shingle_lease_nospace_filter = token_filter(
    "shingle_lease",
    type="shingle",
    min_shingle_size=2,
    max_shingle_size=5,
    output_unigrams=False,
    token_separator="",  # No space separation between phrases
)

# ================================
# Completion analyzer
# ================================

# Used for search suggestions and autocomplete functionality
# For example input: "Machine Learning"
# -> Tokenization: ["machine", "learning"]
# -> Generate shingles: ["machine learning"]
# Suitable for: search box autocomplete, suggestion features
completion_analyzer = analyzer(
    "completion_analyzer",
    tokenizer="standard",  # Standard tokenizer, splits by space and punctuation
    filter=["lowercase", "shingle"],  # Convert to lowercase + generate phrases
)

# ================================
# Edge N-gram analyzer
# ================================

# Edge N-gram tokenizer - generates substrings starting from the beginning of the word
# For example: "elasticsearch" -> ["e", "el", "ela", "elas", ..., "elasticsearch"]
edge_tokenizer = tokenizer(
    "edge_tokenizer",
    type="edge_ngram",
    min_gram=1,  # Minimum character count
    max_gram=20,  # Maximum character count
)

# Edge N-gram analyzer - used for prefix matching search
# For example: "Elasticsearch" -> ["e", "el", "ela", "elas", "elast", ..., "elasticsearch"]
# Suitable for: real-time search during input, prefix matching
edge_analyzer = analyzer(
    "edge_analyzer",
    tokenizer=edge_tokenizer,
    filter=["lowercase"],  # Convert to lowercase
)

# ================================
# Keyword analyzer
# ================================

# Lowercase keyword analyzer - treats entire input as a single token but converts to lowercase
# For example: "Hello World" -> ["hello world"] (as one complete token)
# Suitable for: exact match, status fields, category fields
lower_keyword_analyzer = analyzer(
    "lowercase_keyword",
    tokenizer="keyword",  # No tokenization, entire input as one token
    filter=["lowercase"],  # Convert to lowercase
)

# ================================
# Normalizer
# ================================

# Lowercase normalizer - used for normalization of keyword fields
# For example: "Hello World" -> "hello world"
# Unlike analyzer, normalizer is used for keyword fields and does not perform tokenization
# Suitable for: case normalization during sorting and aggregation
lower_normalizer = normalizer(
    "lower_normalizer",
    char_filter=[],  # No character filters
    filter=["lowercase"],  # Only convert to lowercase
)

# ================================
# English stemming analyzer
# ================================

# English Snowball stem filter - reduces English words to their root form
# For example: "running", "runs", "ran" -> "run"
#      "better", "good" -> "good", "better" (irregular forms require special handling)
snow_en_filter = token_filter(
    "snow_filter", type="snowball", language="English"  # English stemming
)

# English stemming analyzer - used for semantic search on English text
# For example: "I am running quickly"
# -> Tokenization: ["i", "am", "running", "quickly"]
# -> Stemming: ["i", "am", "run", "quick"]
# Suitable for: English document search, improving recall
snow_en_analyzer = analyzer(
    "snow_analyzer",
    tokenizer="standard",  # Standard tokenization
    filter=["lowercase", snow_en_filter],  # Lowercase + stemming
)

# ================================
# Shingle analyzer - with space version
# ================================

# Space-based shingle analyzer - generates phrases after splitting by space
# For example: "hello world test case"
# -> Tokenization: ["hello", "world", "test", "case"]
# -> Shingles: ["hello world", "world test", "test case", "hello world test", ...]
# Suitable for: phrase search, multi-word matching
shingle_space_analyzer = analyzer(
    "shingle_space_analyzer",
    tokenizer="whitespace",  # Split by whitespace
    filter=["lowercase", shingle_lease_filter],  # Lowercase + generate phrases
)

# ================================
# Shingle analyzer - without space version
# ================================

# No-space shingle analyzer - suitable for Chinese or continuous character processing
# For example: "hello-world_test"
# -> word_delimiter_graph decomposition: ["hello", "world", "test"]
# -> no-space shingle: ["helloworld", "worldtest", "helloworldtest"]
# Suitable for: Chinese text, code search, compound word processing
shingle_nospace_analyzer = analyzer(
    "shingle_nospace_analyzer",
    tokenizer="keyword",  # No tokenization, keep original input
    filter=[
        "lowercase",  # Convert to lowercase
        "word_delimiter_graph",  # Split by delimiters (-,_ etc.)
        shingle_lease_nospace_filter,  # Generate no-space phrases
    ],
)

# ================================
# Pre-tokenized content analyzer - for BM25 search on application-layer tokenized text
# ================================

# Pre-tokenized text BM25 analyzer - used for BM25 search on content already tokenized at application layer
# Application layer handles jieba tokenization, ES performs whitespace tokenization and stopword filtering to improve search quality
# For example: application input "我 今天 去了 北京大学" (space-separated tokenization result)
# -> Tokenization: ["我", "今天", "去了", "北京大学"]
# -> Stopword filtering: ["今天", "去了", "北京大学"] (assuming "我" is a stopword)
# Suitable for: Chinese document BM25 search, relevance search on pre-tokenized content
whitespace_lowercase_trim_stop_analyzer = analyzer(
    "whitespace_lowercase_trim_stop_analyzer",
    tokenizer="whitespace",  # Whitespace tokenization for pre-tokenized content
    filter=[
        "lowercase",  # Convert to lowercase
        "trim",  # Remove leading/trailing whitespace
        "stop",  # Stopword filtering to improve search relevance
    ],
)
