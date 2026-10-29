"""The EverOS demo colour palette, shared by the app shell and the sphere.

One definition per colour: the Textual CSS, the Rich copy and the sphere
renderer all read their colours here, so a palette change touches one module.
Colours are plain ``#RRGGBB`` strings because the sphere blends them
channel-wise.
"""

from __future__ import annotations

EVEROS_BLACK = "#1D1C18"
EVEROS_SURFACE = "#24231E"
EVEROS_SURFACE_RAISED = "#31302B"
EVEROS_INK = "#F5EDDC"
EVEROS_MUTED = "#918C80"
EVEROS_BORDER = "#5A5549"

EVEROS_YELLOW = "#F9B91C"
EVEROS_YELLOW_SOFT = "#F6C23B"
EVEROS_YELLOW_PALE = "#FFD267"
EVEROS_AMBER_DIM = "#4A3D20"
EVEROS_AMBER = "#8B763F"
EVEROS_GOLD_SHADOW = "#61522F"
EVEROS_GOLD_DEEP = "#76612F"
EVEROS_GOLD_DARK = "#8C6D2B"
EVEROS_GOLD_MID = "#A97D25"
EVEROS_GOLD_WARM = "#C48E20"
EVEROS_GOLD_LIGHT = "#DDA21E"
EVEROS_CYAN = "#F5EDDC"
EVEROS_GREEN = "#D8CDAF"
EVEROS_ORANGE = "#C09525"
EVEROS_FIELD_BACKGROUND = "#24231E"
