"""Generate reproducible cached LLM context features for upcoming fixtures.

Workflow:
  1. init-sources: create one source-note markdown file per fixture.
  2. Fill each source note with short pre-match facts and URLs.
  3. generate: call the OpenAI Responses API with the saved prompt/schema.
  4. validate: check the resulting jsonl/csv context file.

The model output is used only as structured context. It never predicts
probabilities or match outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import TODAY, build_features, load_data


RAW_COLUMNS = [
    "home_absence_severity",
    "away_absence_severity",
    "home_lineup_uncertainty",
    "away_lineup_uncertainty",
    "home_rotation_risk",
    "away_rotation_risk",
    "home_tactical_edge",
    "away_tactical_edge",
    "llm_confidence",
]

DIFF_COLUMNS = [
    "absence_diff",
    "lineup_uncertainty_diff",
    "rotation_risk_diff",
    "tactical_edge_diff",
]

EXTRA_COLUMNS = [
    "evidence_summary",
    "source_quality",
    "prompt_sha256",
    "schema_sha256",
    "source_sha256",
    "model",
    "generated_at_utc",
    "response_id",
]

OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    *RAW_COLUMNS,
    *DIFF_COLUMNS,
    *EXTRA_COLUMNS,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def fixture_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["date"], row["home_team"], row["away_team"]


def source_path(source_dir: Path, row: dict[str, Any]) -> Path:
    date = row["date"]
    home = slug(row["home_team"])
    away = slug(row["away_team"])
    return source_dir / f"{date}_{home}_vs_{away}.md"


def load_fixtures(from_date: str | None, limit: int | None, refresh: bool) -> list[dict[str, Any]]:
    start = pd.Timestamp(from_date) if from_date else TODAY
    df = load_data(refresh=refresh)
    feats = build_features(df)
    future = feats[
        feats["home_score"].isna() & (feats["date"] >= start)
    ].sort_values("date")
    if limit is not None:
        future = future.head(limit)

    cols = ["date", "home_team", "away_team", "tournament"]
    rows = future[cols].copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    return rows.to_dict(orient="records")


def source_template(row: dict[str, Any]) -> str:
    now = utc_now()
    return f"""---
date: {row["date"]}
home_team: {row["home_team"]}
away_team: {row["away_team"]}
tournament: {row.get("tournament", "")}
status: todo
created_at_utc: {now}
---

# {row["home_team"]} vs {row["away_team"]} source note

Use this file as the reproducible input to the LLM extractor.

Rules:
- Include only information available before kickoff.
- Use short factual summaries, not copied article text.
- Include source URLs and publication times when available.
- Change `status: todo` to `status: ready` before running generation.

## Source URLs
- TODO: URL, publisher, published_at, retrieved_at

## Home team: {row["home_team"]}

### Absences and fitness
- TODO

### Lineup uncertainty
- TODO

### Rotation risk
- TODO

### Tactical notes
- TODO

## Away team: {row["away_team"]}

### Absences and fitness
- TODO

### Lineup uncertainty
- TODO

### Rotation risk
- TODO

### Tactical notes
- TODO

## Other context
- TODO
"""


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    raw = text[4:end]
    body = text[end + 5 :]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def init_sources(args: argparse.Namespace) -> int:
    fixtures = load_fixtures(args.from_date, args.limit, args.refresh)
    made = 0
    for row in fixtures:
        path = source_path(args.source_dir, row)
        if path.exists() and not args.force:
            continue
        write_text(path, source_template(row))
        made += 1
    print(f"Fixtures: {len(fixtures)}")
    print(f"Source notes created/updated: {made}")
    print(f"Source dir: {args.source_dir}")
    return 0


def list_fixtures(args: argparse.Namespace) -> int:
    fixtures = load_fixtures(args.from_date, args.limit, args.refresh)
    for row in fixtures:
        print(f"{row['date']}  {row['home_team']} vs {row['away_team']}  ({row['tournament']})")
    print(f"\n{len(fixtures)} fixture(s)")
    return 0


def build_user_input(meta: dict[str, str], source: str) -> str:
    expected = {
        "date": meta.get("date", ""),
        "home_team": meta.get("home_team", ""),
        "away_team": meta.get("away_team", ""),
        "tournament": meta.get("tournament", ""),
    }
    return (
        "Fixture metadata:\n"
        f"{json.dumps(expected, ensure_ascii=False, sort_keys=True)}\n\n"
        "Source note:\n"
        f"{source}"
    )


def response_body(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    user_input: str,
    temperature: float | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "llm_match_context",
                "strict": True,
                "schema": schema,
            }
        },
    }
    if temperature is not None:
        body["temperature"] = temperature
    return body


def call_openai(body: dict[str, Any], base_url: str, timeout: int) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for generate")

    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if os.environ.get("OPENAI_ORG_ID"):
        headers["OpenAI-Organization"] = os.environ["OPENAI_ORG_ID"]
    if os.environ.get("OPENAI_PROJECT_ID"):
        headers["OpenAI-Project"] = os.environ["OPENAI_PROJECT_ID"]

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    if parts:
        return "\n".join(parts)
    raise ValueError("Could not find output text in OpenAI response")


def compute_diffs(row: dict[str, Any]) -> None:
    row["absence_diff"] = row["away_absence_severity"] - row["home_absence_severity"]
    row["lineup_uncertainty_diff"] = (
        row["away_lineup_uncertainty"] - row["home_lineup_uncertainty"]
    )
    row["rotation_risk_diff"] = row["away_rotation_risk"] - row["home_rotation_risk"]
    row["tactical_edge_diff"] = row["home_tactical_edge"] - row["away_tactical_edge"]


def validate_row(row: dict[str, Any], expected: dict[str, str] | None = None) -> None:
    required = ["date", "home_team", "away_team", *RAW_COLUMNS]
    missing = [col for col in required if col not in row]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if expected:
        for col in ["date", "home_team", "away_team"]:
            if str(row[col]) != str(expected[col]):
                raise ValueError(f"{col} mismatch: expected {expected[col]!r}, got {row[col]!r}")

    int_cols = [col for col in RAW_COLUMNS if col != "llm_confidence"]
    for col in int_cols:
        value = int(row[col])
        if value < 0 or value > 3:
            raise ValueError(f"{col} out of range: {value}")
        row[col] = value

    conf = float(row["llm_confidence"])
    if conf < 0 or conf > 1:
        raise ValueError(f"llm_confidence out of range: {conf}")
    row["llm_confidence"] = conf


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_text(path, text)


def upsert_rows(path: Path, new_rows: list[dict[str, Any]]) -> None:
    existing = read_jsonl(path)
    keyed = {fixture_key(row): row for row in existing}
    for row in new_rows:
        keyed[fixture_key(row)] = row
    rows = sorted(keyed.values(), key=lambda r: fixture_key(r))
    write_jsonl(path, rows)


def export_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = read_jsonl(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def append_audit(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_is_ready(meta: dict[str, str], source: str, include_todo: bool) -> bool:
    status = meta.get("status", "").strip().lower()
    if include_todo:
        return True
    if status not in {"ready", "reviewed"}:
        return False
    if "TODO" in source:
        return False
    return True


def generate(args: argparse.Namespace) -> int:
    prompt = read_text(args.prompt)
    schema = json.loads(read_text(args.schema))
    prompt_hash = sha256_text(prompt)
    schema_hash = sha256_text(json.dumps(schema, sort_keys=True))

    fixtures = load_fixtures(args.from_date, args.limit, args.refresh)
    generated: list[dict[str, Any]] = []
    skipped = 0

    for i, fixture in enumerate(fixtures, start=1):
        path = source_path(args.source_dir, fixture)
        if not path.exists():
            print(f"[skip] missing source note: {path}")
            skipped += 1
            continue

        source = read_text(path)
        source_hash = sha256_text(source)
        meta, _ = parse_front_matter(source)
        if not source_is_ready(meta, source, args.include_todo):
            print(f"[skip] not ready: {path}")
            skipped += 1
            continue

        expected = {
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
        }
        user_input = build_user_input({**meta, **expected}, source)
        body = response_body(args.model, prompt, schema, user_input, args.temperature)

        print(f"[{i}/{len(fixtures)}] {fixture['date']} {fixture['home_team']} vs {fixture['away_team']}")
        if args.dry_run:
            print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])
            continue

        response = call_openai(body, args.base_url, args.timeout)
        output_text = extract_output_text(response)
        row = json.loads(output_text)
        validate_row(row, expected)
        compute_diffs(row)
        row.update(
            {
                "prompt_sha256": prompt_hash,
                "schema_sha256": schema_hash,
                "source_sha256": source_hash,
                "model": args.model,
                "generated_at_utc": utc_now(),
                "response_id": response.get("id", ""),
            }
        )
        generated.append(row)

        append_audit(
            args.audit,
            {
                "date": fixture["date"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "generated_at_utc": row["generated_at_utc"],
                "model": args.model,
                "prompt_path": str(args.prompt),
                "schema_path": str(args.schema),
                "source_path": str(path),
                "prompt_sha256": prompt_hash,
                "schema_sha256": schema_hash,
                "source_sha256": source_hash,
                "request": body,
                "response": response,
            },
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    if generated and not args.dry_run:
        upsert_rows(args.out, generated)
        export_csv(args.out, args.csv)

    print(f"Generated: {len(generated)}")
    print(f"Skipped: {skipped}")
    print(f"JSONL: {args.out}")
    print(f"CSV: {args.csv}")
    print(f"Audit: {args.audit}")
    return 0


def validate_context(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    for row in rows:
        validate_row(row)
        compute_diffs(row)
    print(f"Valid rows: {len(rows)}")
    return 0


def add_common_fixture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-date", default=None, help="Fixture date lower bound, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of fixtures")
    parser.add_argument("--refresh", action="store_true", help="Refresh source results.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-sources", help="Create source note templates")
    add_common_fixture_args(init)
    init.add_argument("--source-dir", type=Path, default=Path("llm_sources"))
    init.add_argument("--force", action="store_true", help="Overwrite existing source notes")
    init.set_defaults(func=init_sources)

    listing = sub.add_parser("list-fixtures", help="List upcoming fixtures")
    add_common_fixture_args(listing)
    listing.set_defaults(func=list_fixtures)

    gen = sub.add_parser("generate", help="Generate llm_context jsonl/csv from ready source notes")
    add_common_fixture_args(gen)
    gen.add_argument("--source-dir", type=Path, default=Path("llm_sources"))
    gen.add_argument("--prompt", type=Path, default=Path("prompts/llm_context_extractor.md"))
    gen.add_argument("--schema", type=Path, default=Path("schemas/llm_context.schema.json"))
    gen.add_argument("--out", type=Path, default=Path("llm_context.jsonl"))
    gen.add_argument("--csv", type=Path, default=Path("llm_context.csv"))
    gen.add_argument("--audit", type=Path, default=Path("llm_context_audit.jsonl"))
    gen.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    gen.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    gen.add_argument("--timeout", type=int, default=120)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--sleep", type=float, default=0.0)
    gen.add_argument("--include-todo", action="store_true", help="Allow source notes still marked todo")
    gen.add_argument("--dry-run", action="store_true", help="Print request payloads without API calls")
    gen.set_defaults(func=generate)

    val = sub.add_parser("validate", help="Validate an llm_context JSONL file")
    val.add_argument("path", type=Path, nargs="?", default=Path("llm_context.jsonl"))
    val.set_defaults(func=validate_context)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
