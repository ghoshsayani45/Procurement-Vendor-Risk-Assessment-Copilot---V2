"""Input and output schemas of the five tools. Strict: unknown fields are rejected."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

VendorId = Field(pattern=r"^V\d{3}$")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _upper_ids(v):
    return [x.strip().upper() for x in v] if isinstance(v, list) else v


# ------------------------------------------------------------------ shared pieces
class SourceRef(_M):
    """A citable source for a computed value: the document (and evidence record) it comes from."""
    doc_id: str
    evidence_id: Optional[str] = None
    title: Optional[str] = None


class ToolFinding(_M):
    code: str
    dimension: str
    points: int
    severity: str
    statement: str
    sources: list[SourceRef]
    needs_legal_review: bool = False


# ------------------------------------------------------------------ vendor_search
class VendorSearchInput(_M):
    query: Optional[str] = Field(default=None, max_length=200, description="vendor name or part of it, e.g. 'Payvanta'")
    vendor_ids: Optional[list[str]] = Field(default=None, max_length=10)
    category: Optional[str] = Field(default=None, max_length=100, description="e.g. 'HRMS', 'cloud hosting'")
    comparison_group: Optional[str] = Field(default=None, max_length=40, description="list the vendors of one comparison group")
    include_record: bool = Field(default=False, description="also return the full vendor record (evidence on file, contract, ...)")
    limit: int = Field(default=5, ge=1, le=35)

    _ids = field_validator("vendor_ids", mode="before")(_upper_ids)


class VendorSummary(_M):
    vendor_id: str
    legal_name: str
    category: str
    comparison_group: str
    status: str
    review_type: str
    data_access: str
    business_critical: bool
    annual_value_inr: int
    business_owner: str
    contract_start: Optional[date] = None
    contract_end: Optional[date] = None
    match_score: float = 1.0


class VendorSearchOutput(_M):
    resolution: Literal["unique", "ambiguous", "none", "listed"]
    matches: list[VendorSummary]
    records: dict[str, dict] = Field(default_factory=dict)
    documents: dict[str, list[dict]] = Field(default_factory=dict)
    note: str = ""


# ------------------------------------------------------------------ document_search
class DocumentSearchInput(_M):
    query: str = Field(min_length=3, max_length=500)
    vendor_ids: Optional[list[str]] = Field(default=None, max_length=3)
    doc_types: Optional[list[str]] = Field(default=None, max_length=10)
    include_superseded: bool = True
    k: int = Field(default=6, ge=1, le=6)

    _ids = field_validator("vendor_ids", mode="before")(_upper_ids)


class PassageOut(_M):
    passage_id: str
    doc_id: str
    title: str
    vendor_id: Optional[str]
    evidence_type: str
    document_date: Optional[str]
    status: str
    heading: str = ""
    text: str
    score: float
    rerank_score: Optional[float] = None
    flags: list[str] = Field(default_factory=list)       # possible_injection:<pattern>, not_evidence, superseded, pii_redacted
    citation: dict


class DocumentSearchOutput(_M):
    mode: Literal["hybrid", "bm25"]
    passages: list[PassageOut]
    injection_flags: int = 0
    note: str = ""


# ------------------------------------------------------------------ compliance_checker / financial_risk_calculator
class VendorAsOfInput(_M):
    vendor_id: str = VendorId
    as_of: Optional[date] = Field(default=None, description="review date; default REVIEW_AS_OF (2026-11-01)")

    @field_validator("vendor_id", mode="before")
    @classmethod
    def _up(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class ChecklistRow(_M):
    item: str
    required: bool
    status: str                      # VERIFIED | EXPIRING_SOON | EXPIRED | MISSING | CONTRADICTORY | NOT_REQUIRED ...
    evidence_ids: list[str]
    sources: list[SourceRef]
    reason: str


class ComplianceOutput(_M):
    vendor_id: str
    legal_name: str
    as_of: date
    tier: Literal[1, 2, 3]
    tier_reasons: list[str]
    checklist: list[ChecklistRow]
    summary: dict[str, list[str]]    # status -> items
    contradictions: list[str]
    findings: list[ToolFinding]      # security, data protection, contract and governance findings
    knockouts: list[str]
    rubric_version: str
    document_warnings: list[dict] = Field(default_factory=list)   # vendor documents with instruction-like text aimed at AI
    note: str = "Statuses are computed from the vendor's records on file. MISSING or EXPIRED evidence is never treated as compliant."


class FinancialYearOut(_M):
    fiscal_year: str
    revenue_inr: int
    revenue_growth_pct: float
    operating_margin_pct: float
    current_ratio: float
    debt_to_equity: float
    source: SourceRef


class DimensionOut(_M):
    dimension: str
    points: int
    weight: float
    applicable: bool
    weighted: float
    triggered_rules: list[str]


class FinancialRiskOutput(_M):
    vendor_id: str
    legal_name: str
    as_of: date
    tier: Literal[1, 2, 3]
    financial_years: list[FinancialYearOut]
    statements_required: int
    statements_on_file: int
    performance: Optional[dict]
    findings: list[ToolFinding]      # financial health and operational performance findings
    dimensions: list[DimensionOut]
    overall: float
    raw_band: str
    band: str
    knockouts: list[str]
    recommendation: str
    conditions: list[str]
    required_approvers: list[str]
    rubric_version: str
    note: str = "Deterministic: computed by the rubric engine from the records on file; the LLM must not recompute it."


# ------------------------------------------------------------------ vendor_compare
class VendorCompareInput(_M):
    vendor_ids: list[str] = Field(min_length=2, max_length=3)
    as_of: Optional[date] = None

    _ids = field_validator("vendor_ids", mode="before")(_upper_ids)


class CompareRow(_M):
    rank: int
    vendor_id: str
    legal_name: str
    comparison_group: str
    tier: Literal[1, 2, 3]
    overall: float
    band: str
    knockouts: list[str]
    dimension_points: dict[str, int]
    key_findings: list[str]
    recommendation: str
    annual_value_inr: int


class VendorCompareOutput(_M):
    rows: list[CompareRow]
    ranking: list[str]
    comparable: bool
    caveats: list[str]
    ranking_rule: str = "Lower risk first: band (LOW < MEDIUM < HIGH < CRITICAL), then overall score, then vendor id."
