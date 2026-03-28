"""
video/transitions.py
Builds FFmpeg xfade filter chains for smooth clip transitions.
"""

from __future__ import annotations

from typing import List, Tuple


XFADE_TRANSITIONS = [
    "fade", "fadeblack", "fadewhite",
    "slideleft", "slideright", "slideup",
    "dissolve", "wipeleft", "wiperight",
]


def build_xfade_filter(
    n_clips: int,
    durations: List[float],
    transition_sec: float = 0.4,
    transition_name: str = "fade",
) -> Tuple[str, str]:
    """
    Build an FFmpeg filter_complex string for N clips with xfade transitions.

    Returns:
        (filter_str, output_label)
        e.g. ("[0:v][1:v]xfade=...[xf1];[xf1][2:v]xfade=...[vout]", "[vout]")
    """
    if n_clips == 1:
        return "[0:v]null[vout]", "[vout]"

    td = min(transition_sec, 0.3)  # hard cap for reliability
    parts: List[str] = []
    offset = 0.0

    for i in range(n_clips - 1):
        offset += max(0.1, durations[i] - td)
        a   = f"[{i}:v]"   if i == 0 else f"[xf{i}]"
        b   = f"[{i+1}:v]"
        out = f"[xf{i+1}]" if i < n_clips - 2 else "[vout]"
        parts.append(
            f"{a}{b}xfade=transition={transition_name}"
            f":duration={td:.3f}:offset={offset:.3f}{out}"
        )

    return ";".join(parts), "[vout]"


def simple_concat_filter(n_clips: int) -> Tuple[str, str]:
    """
    Build a concat filter as fallback when xfade fails.
    """
    inputs = "".join(f"[{i}:v]" for i in range(n_clips))
    return f"{inputs}concat=n={n_clips}:v=1:a=0[vout]", "[vout]"


def ken_burns_filter(
    duration: float,
    fps: int,
    width: int,
    height: int,
    effect: str = "zoom_in",
) -> str:
    """
    Build an FFmpeg zoompan filter string for a Ken Burns effect.

    Effects:
        zoom_in, zoom_out, pan_right, pan_left
    """
    n = max(1, int(duration * fps))
    s = f"{width}x{height}"
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"

    effects = {
        "zoom_in":   f"zoompan=z='min(zoom+0.001,1.25)':d={n}:x='{cx}':y='{cy}':s={s}:fps={fps}",
        "zoom_out":  f"zoompan=z='if(lte(zoom,1.0),1.25,max(1.0,zoom-0.001))':d={n}:x='{cx}':y='{cy}':s={s}:fps={fps}",
        "pan_right": f"zoompan=z='1.15':d={n}:x='min(x+0.5,iw*(1-1/zoom))':y='{cy}':s={s}:fps={fps}",
        "pan_left":  f"zoompan=z='1.15':d={n}:x='max(x-0.5,0)':y='{cy}':s={s}:fps={fps}",
    }
    return effects.get(effect, effects["zoom_in"])
