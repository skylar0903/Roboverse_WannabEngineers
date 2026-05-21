from __future__ import annotations

import subprocess
from typing import Iterable, List


def list_gz_topics() -> List[str]:
    """Return Gazebo Transport topics using whichever CLI exists in the VM."""
    commands = [["gz", "topic", "-l"], ["ign", "topic", "-l"]]
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            topics = [line.strip() for line in out.splitlines() if line.strip()]
            if topics:
                return sorted(set(topics))
        except Exception:
            continue
    return []


def choose_topic(
    configured: str,
    preferred: Iterable[str],
    must_contain_any: Iterable[str],
    avoid_contains_any: Iterable[str] = (),
) -> str | None:
    """Choose a topic.

    If configured is not AUTO, return configured.
    Otherwise list gz topics and score candidates by keyword/preferred match.
    """
    if configured and configured.upper() != "AUTO":
        return configured

    topics = list_gz_topics()
    if not topics:
        return None

    preferred_set = set(preferred)
    required = [s.lower() for s in must_contain_any]
    avoid = [s.lower() for s in avoid_contains_any]

    scored = []
    for t in topics:
        low = t.lower()
        if required and not any(k in low for k in required):
            continue
        if avoid and any(k in low for k in avoid):
            continue
        score = 0
        if t in preferred_set:
            score += 100
        if "image" in low:
            score += 10
        if "camera" in low:
            score += 5
        if "depth" in low:
            score += 8
        if "mono" in low:
            score += 2
        scored.append((score, t))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def print_topic_help() -> None:
    print("\nIf topic AUTO fails, run this in another terminal:")
    print("  gz topic -l")
    print("Then copy the RGB image topic into config.RGB_TOPIC")
    print("and the depth image topic into config.DEPTH_TOPIC.\n")
