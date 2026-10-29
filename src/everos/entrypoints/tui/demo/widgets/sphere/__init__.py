"""Dot-sphere primitives for the EverOS demo TUI.

The Textual app consumes these pure rendering primitives so the animated
surface stays testable without standing up a terminal UI. Build a frame for a
named state with ``build_dot_sphere``, ease between two with
``blend_dot_sphere_frames`` and turn one into Rich text with
``render_dot_sphere_text``.
"""

from everos.entrypoints.tui.demo.widgets.sphere._models import (
    EXTRACT_BRANCH_COUNT as EXTRACT_BRANCH_COUNT,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    SHARED_EDGE_INNER_RADIUS as SHARED_EDGE_INNER_RADIUS,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    SPHERE_STATES as SPHERE_STATES,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import DotCell as DotCell
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    DotSphereFrame as DotSphereFrame,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    SphereState as SphereState,
)
from everos.entrypoints.tui.demo.widgets.sphere._render import (
    blend_dot_sphere_frames as blend_dot_sphere_frames,
)
from everos.entrypoints.tui.demo.widgets.sphere._render import (
    build_dot_sphere as build_dot_sphere,
)
from everos.entrypoints.tui.demo.widgets.sphere._render import (
    render_dot_sphere_lines as render_dot_sphere_lines,
)
from everos.entrypoints.tui.demo.widgets.sphere._render import (
    render_dot_sphere_text as render_dot_sphere_text,
)

__all__ = [
    "EXTRACT_BRANCH_COUNT",
    "SHARED_EDGE_INNER_RADIUS",
    "SPHERE_STATES",
    "DotCell",
    "DotSphereFrame",
    "SphereState",
    "blend_dot_sphere_frames",
    "build_dot_sphere",
    "render_dot_sphere_lines",
    "render_dot_sphere_text",
]
