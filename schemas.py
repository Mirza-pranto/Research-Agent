from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    topic: str = Field(..., description="The main research topic")
    objective: str = Field(..., description="The goal of the research")
    questions: List[str] = Field(default_factory=list, description="Key questions to answer")
    sources_needed: List[str] = Field(default_factory=list, description="Suggested source types or topics")


class ExtractedSource(BaseModel):
    title: str = Field(..., description="Title of the source")
    url: Optional[str] = Field(default=None, description="Source URL if available")
    snippet: Optional[str] = Field(default=None, description="Short excerpt or summary")
    relevance: Optional[str] = Field(default=None, description="Why the source is relevant")


class ResearchDraft(BaseModel):
    title: str = Field(..., description="Title of the research draft")
    summary: str = Field(..., description="Summary of the findings")
    key_points: List[str] = Field(default_factory=list, description="Main points from the research")
    sources: List[ExtractedSource] = Field(default_factory=list, description="Sources used in the draft")


class FactCheckResult(BaseModel):
    claim: str = Field(..., description="The statement being checked")
    status: str = Field(..., description="Verified, disputed, or unsupported")
    evidence: Optional[str] = Field(default=None, description="Supporting or contradicting evidence")
    confidence: Optional[str] = Field(default=None, description="Confidence level of the fact check")
