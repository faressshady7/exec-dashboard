"""Layer 0 - load and validate settings.yaml.

Every other module calls `load_settings()` and never names a company itself.
If the YAML is wrong (unknown peer, duplicate id, bad formula), this fails
loudly at start-up instead of producing a wrong dashboard later.
"""
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "config" / "settings.yaml"

Category = Literal["cost", "market", "fx", "rates", "macro", "demand", "uncertainty"]


class Project(BaseModel):
    name: str
    start_date: date
    end_date: Optional[date] = None
    currency: str = "USD"
    display_years: int = Field(default=3, ge=1, le=10)
    # Max age of the LATEST observation before it is flagged stale, by publishing frequency.
    # FRED dates a monthly value at the 1st of the month and publishes it weeks later,
    # so "monthly" must allow ~3 months and "quarterly" ~7 months.
    max_age_days: dict[Literal["daily", "monthly", "quarterly", "filings", "news"], int] = {
        "daily": 14, "monthly": 100, "quarterly": 220, "filings": 120, "news": 3,
    }
    sec_user_agent: str = Field(pattern=r".+@.+")   # SEC rejects requests without an email
    database_url: str
    data_dir: str = "data"

    @model_validator(mode="after")
    def dates_in_order(self):
        if self.end_date and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @property
    def end(self) -> date:
        return self.end_date or date.today()

    @property
    def display_start(self) -> date:
        """Rolling window: always the last N years up to today, never a fixed date."""
        e = self.end
        try:
            return e.replace(year=e.year - self.display_years)
        except ValueError:  # 29 Feb
            return e.replace(year=e.year - self.display_years, day=28)


class Company(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z.]{1,6}$")
    name: str
    cik: int = Field(gt=0)
    industry: str
    news_terms: list[str] = []          # exact phrases to search in the news
    caveats: list[str] = []             # comparability warnings shown next to this company's fuel figures

    @property
    def search_terms(self) -> list[str]:
        return self.news_terms or [self.name]

    @property
    def cik10(self) -> str:
        """SEC APIs want the CIK as a 10-digit, zero-padded string."""
        return f"{self.cik:010d}"


class RiskFactor(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope: Literal["general", "industry"] = "general"      # general = affects every company
    label: str
    category: Category
    source: Literal["fred"]
    series: str
    frequency: Literal["daily", "monthly", "quarterly"]
    unit: str
    evidence: str


class NewsFilter(BaseModel):
    headline_terms: dict[str, list[str]] = {}
    tier1: list[str] = []
    tier2: list[str] = []
    blocklist: list[str] = []
    ai_max_tier: int = Field(default=2, ge=1, le=3)


class FinancialMetric(BaseModel):
    label: str
    kind: Literal["duration", "instant"]
    tags: list[str] = Field(min_length=1)


class NewsSettings(BaseModel):
    lookback_days: int = Field(default=30, ge=1, le=90)     # GDELT only searches ~3 months back
    max_articles: int = Field(default=100, ge=1, le=250)    # GDELT max per request
    language: str = "english"
    industry_terms: list[str] = []


class AISettings(BaseModel):
    provider: Literal["gemini", "ollama", "mock"] = "gemini"
    model: str = "gemini-3.8-flash"
    fallback_model: Optional[str] = "gemini-3.5-flash-lite"
    ollama_model: str = "llama3.1:8b"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    thinking_level: Optional[Literal["low", "medium", "high"]] = "low"
    candidates: int = Field(default=12, ge=3, le=40)
    pick: int = Field(default=3, ge=1, le=6)
    weights: dict[Literal["significance", "relevance", "impact"], float] = {
        "significance": 0.5, "relevance": 0.3, "impact": 0.2}
    max_retries: int = Field(default=1, ge=0, le=3)
    flag_terms: dict[str, list[str]] = {}      # wording check: category -> risky words/phrases

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        if abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("ai.weights must sum to 1")
        return self


class DerivedFactor(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope: Literal["general", "industry"] = "general"
    label: str
    formula: str
    unit: str
    evidence: str

    @field_validator("formula")
    @classmethod
    def simple_formula(cls, v: str) -> str:
        # Only "<id> <op> <id>" - keeps formulas safe and auditable
        if not re.fullmatch(r"[a-z][a-z0-9_]*\s*[-+*/]\s*[a-z][a-z0-9_]*", v.strip()):
            raise ValueError(f"formula must look like 'a - b', got: {v!r}")
        return v.strip()

    @property
    def inputs(self) -> list[str]:
        return re.findall(r"[a-z][a-z0-9_]*", self.formula)


class HistoricalScenario(BaseModel):
    """A past episode replayed on today's company (historical stress test).
    Dates are optional: no end = latest data; no start = one year before the end."""
    name: str
    start: Optional[date] = None
    end: Optional[date] = None
    note: str = ""


class MethodSettings(BaseModel):
    """Model thresholds (Step 5). Chosen by the researcher, tested by dashboard.robustness."""
    fuel_min_r2: float = Field(default=0.5, ge=0, le=1)
    rate_min_r2: float = Field(default=0.3, ge=0, le=1)
    unusual_sigma: float = Field(default=2.0, gt=0, le=5)
    exclude_covid: bool = True
    covid_start: date = date(2020, 4, 1)
    covid_end: date = date(2021, 12, 31)

    @property
    def covid(self):
        return (self.covid_start, self.covid_end) if self.exclude_covid else None


class Settings(BaseModel):
    project: Project
    focal_company: str
    peers: list[str]
    companies: list[Company]
    risk_factors: list[RiskFactor]
    derived_factors: list[DerivedFactor] = []
    news: NewsSettings = NewsSettings()
    financial_metrics: dict[str, FinancialMetric] = {}
    ai: AISettings = AISettings()
    news_filter: NewsFilter = NewsFilter()
    method: MethodSettings = MethodSettings()
    scenarios: list[HistoricalScenario] = []

    @model_validator(mode="after")
    def check_references(self):
        tickers = [c.ticker for c in self.companies]
        if len(tickers) != len(set(tickers)):
            raise ValueError("duplicate ticker in companies")
        if self.focal_company not in tickers:
            raise ValueError(f"focal_company {self.focal_company} is not in companies")
        for p in self.peers:
            if p not in tickers:
                raise ValueError(f"peer {p} is not in companies")
        if self.focal_company in self.peers:
            raise ValueError("focal_company cannot also be a peer")

        ids = [f.id for f in self.risk_factors] + [d.id for d in self.derived_factors]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate factor id")
        base_ids = {f.id for f in self.risk_factors}
        for d in self.derived_factors:
            missing = [i for i in d.inputs if i not in base_ids]
            if missing:
                raise ValueError(f"derived factor {d.id} uses unknown factor(s): {missing}")
        return self

    # --- convenience accessors used by later layers ---
    @property
    def database_url(self) -> str:
        return os.environ.get("DATABASE_URL", self.project.database_url)

    @property
    def data_path(self) -> Path:
        p = Path(self.project.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def company(self, ticker: str) -> Company:
        return next(c for c in self.companies if c.ticker == ticker)

    @property
    def focal(self) -> Company:
        return self.company(self.focal_company)

    @property
    def peer_companies(self) -> list[Company]:
        return [self.company(t) for t in self.peers]

    @property
    def in_scope(self) -> list[Company]:
        """Focal first, then peers - the order every chart uses."""
        return [self.focal, *self.peer_companies]

    def factors_by_category(self, category: str) -> list[RiskFactor]:
        return [f for f in self.risk_factors if f.category == category]


def load_settings(path: str | Path = DEFAULT_PATH) -> Settings:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(raw)


if __name__ == "__main__":
    s = load_settings()
    print(f"Project : {s.project.name}")
    print(f"Stored  : {s.project.start_date} -> {s.project.end}  (baseline)")
    print(f"Shown   : {s.project.display_start} -> {s.project.end}  (rolling {s.project.display_years}y)")
    print(f"Focal   : {s.focal.ticker}  {s.focal.name}  CIK {s.focal.cik10}")
    print(f"Peers   : {', '.join(c.ticker for c in s.peer_companies)}")
    print(f"Factors : {len(s.risk_factors)} downloaded + {len(s.derived_factors)} derived")
    for cat in ["cost", "market", "fx", "rates", "macro", "demand"]:
        names = [f.id for f in s.factors_by_category(cat)]
        print(f"  {cat:<7}: {', '.join(names)}")
    todo = [f.id for f in s.risk_factors if f.evidence.startswith("TODO")]
    if todo:
        print(f"Needs a source: {', '.join(todo)}")
