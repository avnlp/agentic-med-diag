"""Message model for multi-turn agent input."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """A single conversation turn fed to the agent.

    Attributes:
        role: Speaker role.
        content: Turn text.
    """

    role: Literal["system", "user", "assistant"] = "user"
    content: str
