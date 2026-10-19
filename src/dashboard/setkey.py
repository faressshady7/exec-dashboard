"""Save the FMP API key into .env (typed hidden, never shown or printed), then test it with one call.

Run:  set_fmp_key.bat     (or: python -m dashboard.setkey)
"""
from __future__ import annotations

import getpass
from datetime import date, timedelta

from dashboard.config import PROJECT_ROOT

ENV = PROJECT_ROOT / ".env"
NAME = "FMP_API_KEY"


def save(key: str, path=ENV) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [l for l in lines if not l.strip().upper().startswith(NAME + "=")]
    lines.append(f"{NAME}={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    key = getpass.getpass("Paste your FMP API key (it stays hidden), then press Enter: ").strip().strip('"')
    if len(key) < 10:
        raise SystemExit("That looks too short - nothing was saved.")
    save(key)
    print(f"Saved to {ENV.name} (git-ignored).")
    from dashboard.bronze import FMP_URL, Fetcher, _mask, parse_fmp_prices
    end = date.today()
    url = FMP_URL.format(symbol="UAL", start=end - timedelta(days=10), end=end, key=key)
    try:
        n, latest = parse_fmp_prices(Fetcher("exec-dashboard thesis", retries=1).get(url), "UAL")
        print(f"Key works: {n} trading days for UAL, latest {latest}.")
    except Exception as e:
        print("Key saved, but the test call failed:", _mask(str(e), key))


if __name__ == "__main__":
    main()
