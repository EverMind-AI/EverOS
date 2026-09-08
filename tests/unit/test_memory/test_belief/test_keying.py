"""Pins belief-key derivation: what competes, what must not, and the bias.

The contract worth defending is asymmetric. Missing a link costs nothing
that was not already lost — the facts stay side by side, which is where
EverOS is without this module. Inventing a link costs a true fact its
place, silently. So these tests pin the *precision* side hardest.
"""

from __future__ import annotations

from everos.memory.belief.keying import BeliefKeyer, signature


def test_signature_drops_the_value_and_keeps_the_topic() -> None:
    """Candidates of one belief differ exactly where the signature ignores."""
    cheap = signature("I got pre-approved for $350,000 from Wells Fargo")
    dear = signature("I got pre-approved for $400,000 from Wells Fargo")

    assert cheap == dear
    assert "wells" in cheap
    assert "350,000" not in cheap


def test_a_changed_quantity_lands_on_the_same_belief() -> None:
    keyer = BeliefKeyer()

    first = keyer.key_for("I've tried three different Korean restaurants in my city")
    second = keyer.key_for("I've tried four different Korean restaurants in my city")

    assert first == second


def test_unrelated_facts_do_not_compete() -> None:
    keyer = BeliefKeyer()

    running = keyer.key_for("I set a personal best in the charity 5K run of 27:12")
    mortgage = keyer.key_for("I got pre-approved for $350,000 from Wells Fargo")

    assert running != mortgage


def test_scopes_never_compete() -> None:
    """Two users saying the same sentence hold two separate beliefs."""
    keyer = BeliefKeyer()

    mine = keyer.key_for("my coffee ratio is 6 ounces per tablespoon", scope="user_a")
    yours = keyer.key_for("my coffee ratio is 6 ounces per tablespoon", scope="user_b")

    assert mine != yours
    assert mine.startswith("user_a:")


def test_keys_are_stable_across_keyer_instances() -> None:
    """A key minted today must still name the same belief after a restart."""
    fact = "my coffee ratio is 6 ounces per tablespoon"

    assert BeliefKeyer().key_for(fact, scope="u") == BeliefKeyer().key_for(
        fact, scope="u"
    )


def test_a_belief_signature_does_not_widen_as_members_join() -> None:
    """Anti-drift: a belief must not grow into its neighbour and swallow it.

    The two 5K facts join one belief. If joining widened that belief's
    signature to the union of its members, the vocabulary it matches
    against would keep growing and eventually cover the unrelated fact.
    """
    keyer = BeliefKeyer()
    key = keyer.key_for("my personal best in the charity 5K run was 27:12")
    keyer.key_for("I want to beat my personal best 5K time of 25:50 this year")

    unrelated = keyer.key_for(
        "I finished my fifth issue of National Geographic about the Amazon"
    )

    assert unrelated != key


def test_an_empty_signature_still_gets_its_own_key() -> None:
    """A fact of pure stopwords and numbers competes with nothing."""
    keyer = BeliefKeyer()

    first = keyer.key_for("it is 5")
    second = keyer.key_for("it is 6")

    assert first != second


def test_threshold_trades_recall_for_precision_in_the_stated_direction() -> None:
    near_miss = "the standing desk in my home office is 48 inches wide"
    other = "my home office chair is 5 years old"

    permissive = BeliefKeyer(threshold=0.05)
    strict = BeliefKeyer(threshold=0.9)

    assert permissive.key_for(near_miss) == permissive.key_for(other)
    assert strict.key_for(near_miss) != strict.key_for(other)
