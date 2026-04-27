import os


VECTORIZE_DIMENSIONS = int(os.getenv("VECTORIZE_DIMENSIONS", "1024"))

EXTRACT_SCENES = ("boundary", "extraction", "profile")
