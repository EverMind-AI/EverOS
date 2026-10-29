"""The canned knowledge scripts satisfy the real extractor they stand in for.

Each script under ``tests/fakes/llm_scripts/knowledge`` is held to the strict
reply model the everalgo knowledge extractor requests, and the default set is
fed to the real extractor, so a script that drifted fails here rather than
inside the tier-3 integration suite.
"""

from __future__ import annotations

import json

import pytest
from everalgo.knowledge import KnowledgeExtractor
from everalgo.knowledge import _replies as replies
from everalgo.llm import parse_reply
from everalgo.types import CategorySpec, ParsedContent

from tests.fakes import KNOWLEDGE_OPERATORS, knowledge_llm, knowledge_script


@pytest.mark.parametrize(("operator", "reply"), sorted(KNOWLEDGE_OPERATORS.items()))
def test_every_script_matches_its_reply_model(operator: str, reply: str) -> None:
    parse_reply(getattr(replies, reply), knowledge_script(operator), operator=operator)


async def test_default_scripts_extract_one_topic_under_the_root() -> None:
    llm = knowledge_llm()
    memories = await KnowledgeExtractor(llm=llm).aextract(
        ParsedContent(text="Mountain hiking safety guide."),
        doc_id="d_canned00001",
        title="Hiking Guide",
        categories=[CategorySpec(id="Others", description="Anything else.")],
    )

    tree = json.loads(knowledge_script("topic_tree"))
    root, topic = memories
    assert (root.topic_index, root.topic, root.summary) == (
        0,
        "Hiking Guide",
        tree["summary"],
    )
    assert topic.topic == tree["topics"][0]["topic"]
    assert topic.content == "Mountain hiking safety guide."
    assert {m.category_id for m in memories} == {"Others"}
    assert [c.response_format.__name__ for c in llm.calls] == [
        "TopicTreeReply",
        "CategoryReply",
    ]
