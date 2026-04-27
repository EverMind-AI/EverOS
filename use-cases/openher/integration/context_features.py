"""
Neural network context features — showing how EverCore expands
the persona engine's perception from 8D to 12D.

The 4 additional relationship dimensions from EverCore allow the
neural network to produce different behavioral signals depending
on the history between user and persona.

Full source: https://github.com/kellyvv/OpenHer/blob/main/engine/genome/genome_engine.py
"""

# ══════════════════════════════════════════════
# 5D Drive System (internal motivation)
# ══════════════════════════════════════════════

DRIVES = ['connection', 'novelty', 'expression', 'safety', 'play']

# ══════════════════════════════════════════════
# 8D Behavioral Signals (neural network output)
# ══════════════════════════════════════════════

SIGNALS = [
    'directness',     # 0=indirect hints → 1=straight talk
    'vulnerability',  # 0=guarded → 1=emotionally open
    'playfulness',    # 0=serious → 1=playful/teasing
    'initiative',     # 0=reactive → 1=proactive leading
    'depth',          # 0=small talk → 1=deep conversation
    'warmth',         # 0=cold/distant → 1=warm/caring
    'defiance',       # 0=compliant → 1=rebellious/stubborn
    'curiosity',      # 0=indifferent → 1=intensely curious
]

# ══════════════════════════════════════════════
# 12D Context Features (neural network input)
# ══════════════════════════════════════════════

CONTEXT_FEATURES = [
    # ── 8D from Critic LLM (per-turn perception) ──
    'user_emotion',        # -1=negative → 1=positive
    'topic_intimacy',      # 0=professional → 1=intimate
    'time_of_day',         # 0=morning → 1=late night
    'conversation_depth',  # 0=just started → 1=deep conversation
    'user_engagement',     # 0=dismissive → 1=invested
    'conflict_level',      # 0=harmonious → 1=conflict
    'novelty_level',       # 0=routine topic → 1=novel topic
    'user_vulnerability',  # 0=guarded → 1=open

    # ── 4D from EverCore (cross-session relationship) ──
    'relationship_depth',  # 0=stranger → 1=old friend
    'emotional_valence',   # -1=negative history → 1=positive history
    'trust_level',         # 0=no trust → 1=deep trust
    'pending_foresight',   # 0=nothing pending → 1=unresolved concern
]

# Neural network dimensions
N_DRIVES = len(DRIVES)       # 5
N_CONTEXT = len(CONTEXT_FEATURES)  # 12 (8 + 4 from EverCore)
N_SIGNALS = len(SIGNALS)     # 8
RECURRENT_SIZE = 8            # Internal "mood" state
INPUT_SIZE = N_DRIVES + N_CONTEXT + RECURRENT_SIZE  # 5 + 12 + 8 = 25
HIDDEN_SIZE = 24

# Architecture: 25D input → 24D hidden (tanh) → 8D output (sigmoid)
# The 4 EverCore dimensions mean the same neural network produces
# DIFFERENT behavioral signals for strangers vs. old friends,
# even with identical conversation context.
