import copy

import pytest
import yaml
from pydantic import ValidationError

from dashboard.config import DEFAULT_PATH, Settings, load_settings


@pytest.fixture
def raw():
    with open(DEFAULT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_real_config_loads():
    s = load_settings()
    assert s.focal.ticker == "UAL"
    assert s.focal.cik10 == "0000100517"
    assert [c.ticker for c in s.in_scope] == ["UAL", "DAL", "AAL"]


def test_switching_company_needs_only_config(raw):
    # The point of Layer 0: a new focal company is a config change, not a code change
    r = copy.deepcopy(raw)
    r["focal_company"], r["peers"] = "DAL", ["UAL", "AAL"]
    s = Settings.model_validate(r)
    assert s.focal.name.startswith("Delta")


def test_display_window_is_rolling(raw):
    from datetime import date
    r = copy.deepcopy(raw)
    r["project"]["end_date"] = "2026-10-19"
    s = Settings.model_validate(r)
    assert s.project.display_start == date(2023, 10, 19)
    assert s.project.display_start > s.project.start_date   # view is recent, storage is longer


def test_database_url_env_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x@db:5432/y")
    assert load_settings().database_url.endswith("db:5432/y")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(focal_company="XYZ"), "focal_company"),
        (lambda r: r.update(peers=["DAL", "ZZZ"]), "peer ZZZ"),
        (lambda r: r.update(peers=["UAL"]), "cannot also be a peer"),
        (lambda r: r["risk_factors"].append(dict(r["risk_factors"][0])), "duplicate factor"),
        (lambda r: r["derived_factors"][0].update(formula="yield_10y - nope"), "unknown factor"),
        (lambda r: r["derived_factors"][0].update(formula="__import__('os')"), "formula"),
        (lambda r: r["project"].update(end_date="2010-01-01"), "end_date"),
        (lambda r: r["project"].update(sec_user_agent="no email"), "sec_user_agent"),
    ],
)
def test_bad_config_fails_loudly(raw, mutate, message):
    r = copy.deepcopy(raw)
    mutate(r)
    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(r)
