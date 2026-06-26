#!/usr/bin/env python
"""Export site ss000225a0001 to ets/datasets/ssa/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo, ensure_runtime_python

bootstrap_repo(_REPO)
ensure_runtime_python("numpy", "pandas", "psycopg2")

SITE_CODE = "ss000225a0001"
SENSOR_COLS = [
    "s08001",
    "s08002",
    "s08003",
    "s04001",
    "s06001",
    "s06002",
    "s04003",
    "s04004",
    "a01001",
    "a01002",
    "a01006",
    "a01007",
    "a01008",
    "a99943",
    "uv0002",
]

SQL_HOURLY_30D = """
SELECT DISTINCT ON (date_trunc('hour', time))
    date_trunc('hour', time) AS time,
    {cols}
FROM public.farming_realtime_data
WHERE site_code = %s
  AND time >= date_trunc('hour', NOW()) - INTERVAL '30 days'
  AND time < date_trunc('hour', NOW())
ORDER BY date_trunc('hour', time), time DESC
"""

SQL_MINUTE_24H = """
SELECT time, {cols}
FROM public.farming_realtime_data
WHERE site_code = %s
  AND time >= date_trunc('minute', NOW()) - INTERVAL '24 hours'
  AND time < date_trunc('minute', NOW())
ORDER BY time
"""


def _dataset_dir() -> Path:
    return _REPO / "ets" / "datasets" / "ssa"


def _log_dir() -> Path:
    return _REPO / "ets" / "artifacts" / "logs"


def _pg_connect():
    import psycopg2

    host = os.environ.get("FARMING_PG_HOST", "192.168.168.3")
    port = int(os.environ.get("FARMING_PG_PORT", "31010"))
    user = os.environ.get("FARMING_PG_USER", "ssman")
    password = os.environ.get("FARMING_PG_PASSWORD")
    dbname = os.environ.get("FARMING_PG_DB", "iot-manager-server")

    if not password:
        raise SystemExit(
            "FARMING_PG_PASSWORD is not set. "
            "Export credentials via environment variables before running."
        )

    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=30,
    )
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Shanghai'")
    return conn


def _fetch_dataframe(conn, sql: str):
    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(sql, (SITE_CODE,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _atomic_write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_manifest(path: Path, export_type: str, df) -> None:
    time_col = df["time"]
    payload = {
        "export_type": export_type,
        "site_code": SITE_CODE,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "time_min": str(time_col.min()) if len(df) else None,
        "time_max": str(time_col.max()) if len(df) else None,
        "columns": list(df.columns),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def export_hourly_30d(out_dir: Path) -> int:
    cols = ", ".join(SENSOR_COLS)
    sql = SQL_HOURLY_30D.format(cols=cols)
    conn = _pg_connect()
    try:
        df = _fetch_dataframe(conn, sql)
    finally:
        conn.close()

    csv_path = out_dir / "hourly_30d.csv"
    manifest_path = out_dir / "manifest_hourly.json"
    _atomic_write_csv(df, csv_path)
    _write_manifest(manifest_path, "hourly_30d", df)
    print(f"[hourly_30d] wrote {len(df)} rows -> {csv_path}")
    return len(df)


def export_minute_24h(out_dir: Path) -> int:
    cols = ", ".join(SENSOR_COLS)
    sql = SQL_MINUTE_24H.format(cols=cols)
    conn = _pg_connect()
    try:
        df = _fetch_dataframe(conn, sql)
    finally:
        conn.close()

    csv_path = out_dir / "minute_24h.csv"
    manifest_path = out_dir / "manifest_minute.json"
    _atomic_write_csv(df, csv_path)
    _write_manifest(manifest_path, "minute_24h", df)
    print(f"[minute_24h] wrote {len(df)} rows -> {csv_path}")
    return len(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync ssa dataset from PostgreSQL.")
    parser.add_argument(
        "--export",
        choices=["hourly_30d", "minute_24h", "all"],
        required=True,
        help="Which dataset export to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_dataset_dir(),
        help="Output directory for CSV and manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log_dir().mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir.resolve()

    if args.export in ("hourly_30d", "all"):
        export_hourly_30d(out_dir)
    if args.export in ("minute_24h", "all"):
        export_minute_24h(out_dir)


if __name__ == "__main__":
    main()
