from typing import List

import numpy as np

from .modules import Proposal
from langchain_community.tools.human.tool import HumanInputRun


def hints_to_proposals(hints, image_shape) -> List[Proposal]:
    """Convert user-provided points/boxes into Proposal objects."""
    H, W = image_shape[:2]
    proposals: List[Proposal] = []
    for pt in hints.pos_points:
        proposals.append(
            Proposal(
                box=[0, 0, W - 1, H - 1],
                pos_points=[pt],
                neg_points=[],
                module="human_point",
                note="user pos point",
                weight=1.25,
                target="human",
            )
        )
    for pt in hints.neg_points:
        proposals.append(
            Proposal(
                box=[0, 0, W - 1, H - 1],
                pos_points=[],
                neg_points=[pt],
                module="human_neg_point",
                note="user neg point",
                weight=1.10,
                target="human",
            )
        )
    for box in hints.boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        proposals.append(
            Proposal(
                box=[x1, y1, x2, y2],
                pos_points=[[cx, cy]],
                neg_points=[],
                module="human_box",
                note="user box",
                weight=1.30,
                target="human",
            )
        )
    return proposals


def ask_human_for_hints(prompt: str) -> str:
    """
    Blocking human input via LangChain human tool. Returns raw string.
    """
    tool = HumanInputRun()
    return tool.run(prompt)
