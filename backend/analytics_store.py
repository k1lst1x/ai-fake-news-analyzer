from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass
class Summary:
    total_checks: int
    fake_checks: int
    real_checks: int
    fake_share: float
    avg_latency_ms: float


class AnalyticsStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    language TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    country TEXT,
                    label TEXT NOT NULL,
                    fake_prob REAL NOT NULL,
                    real_prob REAL NOT NULL,
                    text_length INTEGER NOT NULL,
                    token_length INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    model_trace TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_checks_created_at ON checks(created_at)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_checks_lang ON checks(language)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_checks_content_type ON checks(content_type)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_checks_label ON checks(label)")
            con.commit()

    def log_check(
        self,
        *,
        language: str,
        content_type: str,
        country: Optional[str],
        label: str,
        fake_prob: float,
        real_prob: float,
        text_length: int,
        token_length: int,
        latency_ms: int,
        model_trace: str,
    ) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO checks (
                    language, content_type, country, label,
                    fake_prob, real_prob, text_length, token_length, latency_ms, model_trace
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    language,
                    content_type,
                    country,
                    label,
                    float(fake_prob),
                    float(real_prob),
                    int(text_length),
                    int(token_length),
                    int(latency_ms),
                    model_trace,
                ),
            )
            con.commit()

    @staticmethod
    def _days_filter_sql(days: int) -> str:
        if days <= 0:
            return ""
        return f" AND created_at >= datetime('now', '-{days} day')"

    @staticmethod
    def _in_clause(column: str, values: Sequence[str]) -> tuple[str, list]:
        if not values:
            return "", []
        placeholders = ",".join("?" for _ in values)
        return f" AND {column} IN ({placeholders})", list(values)

    def summary(self, *, days: int, languages: Sequence[str], content_types: Sequence[str]) -> Summary:
        days_sql = self._days_filter_sql(days)
        lang_sql, lang_args = self._in_clause("language", languages)
        ctype_sql, ctype_args = self._in_clause("content_type", content_types)

        query = (
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN label='fake' THEN 1 ELSE 0 END) AS fake_n, "
            "SUM(CASE WHEN label='real' THEN 1 ELSE 0 END) AS real_n, "
            "AVG(latency_ms) AS avg_latency "
            "FROM checks WHERE 1=1"
            f"{days_sql}{lang_sql}{ctype_sql}"
        )
        args = lang_args + ctype_args
        with self._connect() as con:
            row = con.execute(query, args).fetchone()
        total = int(row["total"] or 0)
        fake_n = int(row["fake_n"] or 0)
        real_n = int(row["real_n"] or 0)
        fake_share = float(fake_n / total) if total else 0.0
        avg_latency = float(row["avg_latency"] or 0.0)
        return Summary(
            total_checks=total,
            fake_checks=fake_n,
            real_checks=real_n,
            fake_share=fake_share,
            avg_latency_ms=avg_latency,
        )

    def timeline(self, *, days: int, languages: Sequence[str], content_types: Sequence[str]) -> pd.DataFrame:
        days_sql = self._days_filter_sql(days)
        lang_sql, lang_args = self._in_clause("language", languages)
        ctype_sql, ctype_args = self._in_clause("content_type", content_types)
        query = (
            "SELECT strftime('%Y-%m-%d', created_at) AS bucket, label, COUNT(*) AS value "
            "FROM checks WHERE 1=1"
            f"{days_sql}{lang_sql}{ctype_sql} "
            "GROUP BY bucket, label ORDER BY bucket"
        )
        args = lang_args + ctype_args
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=args)

    def language_distribution(self, *, days: int, content_types: Sequence[str]) -> pd.DataFrame:
        days_sql = self._days_filter_sql(days)
        ctype_sql, ctype_args = self._in_clause("content_type", content_types)
        query = (
            "SELECT language AS bucket, COUNT(*) AS value "
            "FROM checks WHERE 1=1 "
            f"{days_sql}{ctype_sql} "
            "GROUP BY language ORDER BY value DESC"
        )
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=ctype_args)

    def geo_distribution(self, *, days: int, languages: Sequence[str], content_types: Sequence[str]) -> pd.DataFrame:
        days_sql = self._days_filter_sql(days)
        lang_sql, lang_args = self._in_clause("language", languages)
        ctype_sql, ctype_args = self._in_clause("content_type", content_types)
        query = (
            "SELECT COALESCE(NULLIF(country, ''), 'Unknown') AS bucket, COUNT(*) AS value "
            "FROM checks WHERE 1=1 "
            f"{days_sql}{lang_sql}{ctype_sql} "
            "GROUP BY bucket ORDER BY value DESC LIMIT 20"
        )
        args = lang_args + ctype_args
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=args)

    def export_checks(
        self,
        *,
        days: int,
        languages: Sequence[str],
        content_types: Sequence[str],
    ) -> pd.DataFrame:
        days_sql = self._days_filter_sql(days)
        lang_sql, lang_args = self._in_clause("language", languages)
        ctype_sql, ctype_args = self._in_clause("content_type", content_types)
        query = (
            "SELECT id, created_at, language, content_type, country, label, "
            "fake_prob, real_prob, text_length, token_length, latency_ms, model_trace "
            "FROM checks WHERE 1=1 "
            f"{days_sql}{lang_sql}{ctype_sql} "
            "ORDER BY created_at DESC"
        )
        args = lang_args + ctype_args
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=args)
