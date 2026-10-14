"""Evaluation configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationSettings(BaseSettings):
    """Settings for BAML-backed evaluation."""

    model_config = SettingsConfigDict(env_prefix="EVALUATION_", extra="ignore")

    judge_model: str | None = None
