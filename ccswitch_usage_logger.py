#!/usr/bin/env python3
"""Mirror CC Switch Codex usage rows into a readable desktop log."""

import datetime as _dt
import json
import os
import sqlite3
import time


DB_PATH = os.path.expanduser(os.environ.get("CCSWITCH_DB", "~/.cc-switch/cc-switch.db"))
LOG_PATH = os.path.expanduser(os.environ.get("CCSWITCH_LOG", "~/Desktop/log.log"))
STATE_PATH = os.path.expanduser(os.environ.get("CCSWITCH_LOG_STATE", "~/Desktop/.ccswitch_usage_logger_state.json"))
MODEL = os.environ.get("CCSWITCH_LOG_MODEL", "gpt-5.6-sol")
TARGET_BASE_URL = os.environ.get("CCSWITCH_LOG_BASE_URL", "https://api.hetune.top/v1")
GROUP = os.environ.get("CCSWITCH_LOG_GROUP", "福利组")
KEY_LABEL = os.environ.get("CCSWITCH_LOG_KEY", "my_key")
PLAN_LABEL = os.environ.get("CCSWITCH_LOG_PLAN", "base · $5 / $30/M+1")
POLL_SECONDS = max(1.0, float(os.environ.get("CCSWITCH_LOG_POLL", "2")))
LOG_HEADER = "时间\n令牌\n模型\n流\nTokens\n费用\n耗时\n详情\n\n"


def _number(value, default=0):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _seconds(milliseconds):
    value = _number(milliseconds)
    return f"{value / 1000:.1f}s" if value > 0 else "N/A"


def _speed(output_tokens, duration_ms):
    output = _number(output_tokens)
    duration = _number(duration_ms)
    if output <= 0 or duration <= 0:
        return "N/A"
    return f"{output / (duration / 1000):.0f} t/s"


def format_entry(row):
    """Format one proxy_request_logs row without exposing credentials."""
    timestamp = _dt.datetime.fromtimestamp(_number(row["created_at"])).strftime("%Y-%m-%d %H:%M:%S")
    input_tokens = _number(row["input_tokens"])
    output_tokens = _number(row["output_tokens"])
    cache_tokens = _number(row["cache_read_tokens"])
    duration = row["duration_ms"] or row["latency_ms"]
    stream = "流" if _number(row["is_streaming"]) else "非流"
    cost = str(row["total_cost_usd"] or "0")
    details = (
        f"request_id={row['request_id']}\n"
        f"provider_id={row['provider_id']}\n"
        f"状态={_number(row['status_code'])}"
    )
    return (
        f"{timestamp}\n"
        f"消耗\n"
        f"{KEY_LABEL}\n"
        f"{GROUP}\n"
        f"{row['model'] or MODEL}\n"
        f"{stream}\n"
        f"{_speed(output_tokens, duration)}\n"
        f"{input_tokens:,} / {output_tokens:,}\n"
        f"缓存↓ {cache_tokens:,}\n"
        f"$\n"
        f"{cost}\n"
        f"首字\n"
        f"{_seconds(row['first_token_ms'])}\n"
        f"耗时\n"
        f"{_seconds(duration)}\n\n"
        f"{PLAN_LABEL}\n"
        f"详情\n"
        f"{details}\n\n"
    )


def _load_seen():
    try:
        with open(STATE_PATH, encoding="utf-8") as handle:
            value = json.load(handle)
        # Versions before source-aware tracking stored bare request IDs;
        # those rows came from the proxy stream, so migrate them in place.
        return set(
            item if "::" in item else f"proxy::{item}"
            for item in (value if isinstance(value, list) else [])
            if isinstance(item, str)
        )
    except (OSError, ValueError, TypeError):
        return set()


def _save_seen(seen):
    directory = os.path.dirname(STATE_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = STATE_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(sorted(seen), handle, ensure_ascii=True)
    os.replace(temporary, STATE_PATH)


def _connect():
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    return connection


def _target_provider_ids(connection):
    """Resolve CC Switch provider IDs whose configured endpoint matches."""
    needle = f"%{TARGET_BASE_URL}%"
    try:
        rows = connection.execute(
            "SELECT id FROM providers WHERE app_type = 'codex' AND settings_config LIKE ?",
            (needle,),
        ).fetchall()
    except sqlite3.Error:
        return set()
    return {row[0] for row in rows}


def _session_endpoint_matches():
    """Session-sync rows have no provider ID; verify Codex's active config."""
    config_path = os.path.expanduser("~/.codex/config.toml")
    try:
        with open(config_path, encoding="utf-8") as handle:
            return TARGET_BASE_URL in handle.read()
    except OSError:
        return False


def _fetch(connection, seen):
    query = """
        SELECT request_id, provider_id, model, input_tokens, output_tokens,
               cache_read_tokens, total_cost_usd, latency_ms, first_token_ms,
               duration_ms, status_code, is_streaming, created_at, data_source
        FROM proxy_request_logs
        WHERE app_type = 'codex' AND data_source IN ('proxy', 'codex_session') AND model = ?
        ORDER BY created_at DESC, request_id DESC
    """
    selected = []
    provider_ids = _target_provider_ids(connection)
    for source in ("proxy", "codex_session"):
        source_seen = {key for key in seen if key.startswith(source + "::")}
        source_query = query.replace(
            "data_source IN ('proxy', 'codex_session')", "data_source = ?"
        )
        if source == "proxy":
            if not provider_ids:
                continue
            placeholders = ",".join("?" for _ in provider_ids)
            source_query = source_query.replace(
                "ORDER BY created_at DESC, request_id DESC",
                f"AND provider_id IN ({placeholders}) ORDER BY created_at DESC, request_id DESC",
            )
            source_params = [source, MODEL, *sorted(provider_ids)]
        else:
            if not _session_endpoint_matches():
                continue
            source_query = source_query.replace(
                "ORDER BY created_at DESC, request_id DESC",
                "AND provider_id = '_codex_session' ORDER BY created_at DESC, request_id DESC",
            )
            source_params = [source, MODEL]
        if not source_seen:
            # Bootstrap each source with a useful audit tail, then mark older
            # rows as seen so the first run never replays the whole database.
            rows = connection.execute(source_query + " LIMIT 20", source_params).fetchall()
            all_ids = connection.execute(
                "SELECT request_id FROM proxy_request_logs "
                "WHERE app_type = 'codex' AND data_source = ? AND model = ? "
                + (f"AND provider_id IN ({','.join('?' for _ in provider_ids)})" if source == "proxy" else "AND provider_id = '_codex_session'"),
                ([source, MODEL, *sorted(provider_ids)] if source == "proxy" else [source, MODEL]),
            ).fetchall()
            seen.update(f"{source}::{row[0]}" for row in all_ids)
            selected.extend(rows)
            continue
        rows = connection.execute(source_query, (source, MODEL)).fetchall()
        selected.extend(row for row in rows if f"{source}::{row['request_id']}" not in seen)
    return sorted(selected, key=lambda row: (row["created_at"], row["request_id"]))


def run_once(seen):
    try:
        connection = _connect()
        try:
            rows = _fetch(connection, seen)
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return 0
    if not rows:
        return 0
    directory = os.path.dirname(LOG_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    needs_header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        if needs_header:
            handle.write(LOG_HEADER)
        for row in rows:
            handle.write(format_entry(row))
            seen.add(f"{row['data_source']}::{row['request_id']}")
        handle.flush()
    _save_seen(seen)
    return len(rows)


def main():
    seen = _load_seen()
    while True:
        run_once(seen)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
