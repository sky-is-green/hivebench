#!/usr/bin/env python
"""Local userspace PostgreSQL for the LiteLLM gateway — no root required.

The binaries ship with the ``pgserver`` package; the cluster lives under
``~/.local/share/hivebench-litellm/pgdata``.  Unlike pgserver's own starter
(socket-only, ``listen_addresses=''``), this script runs Postgres on
**127.0.0.1:5433** so Prisma — and therefore LiteLLM — can connect over TCP.

    .venv/bin/python pg.py                     # ensure running; print the URI
    .venv/bin/python pg.py --uri-db litellm    # URI for one database
    .venv/bin/python pg.py --sql "CREATE DATABASE litellm"
    .venv/bin/python pg.py --stop
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

from pgserver._commands import POSTGRES_BIN_PATH

DATA = pathlib.Path.home() / ".local" / "share" / "hivebench-litellm" / "pgdata"
HOST = "127.0.0.1"
PORT = 5433
BIN = pathlib.Path(POSTGRES_BIN_PATH)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN / args[0]), *args[1:]], capture_output=True, text=True)


def _is_running() -> bool:
    return _run("pg_isready", "-h", HOST, "-p", str(PORT)).returncode == 0


def ensure_running() -> None:
    """initdb on first use, then start (or verify) the TCP server."""
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not (DATA / "PG_VERSION").is_file():
        init = _run(
            "initdb", "--auth=trust", "--auth-local=trust", "--encoding=utf8",
            "-U", "postgres", "-D", str(DATA),
        )
        if init.returncode != 0:
            raise SystemExit(init.stderr or init.stdout)
    if _is_running():
        return
    start = _run(
        "pg_ctl", "-D", str(DATA),
        "-o", f"-h {HOST} -p {PORT} -k {DATA}",
        "-l", str(DATA / "server.log"), "-w", "start",
    )
    if start.returncode != 0:
        raise SystemExit(start.stderr or start.stdout)


def stop() -> None:
    _run("pg_ctl", "-D", str(DATA), "-m", "fast", "-w", "stop")


def uri(database: str) -> str:
    return f"postgresql://postgres@{HOST}:{PORT}/{database}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default="", help="run one SQL command as the superuser")
    parser.add_argument("--uri-db", default="", help="print the URI for this database")
    parser.add_argument("--stop", action="store_true", help="stop the server")
    args = parser.parse_args()

    if args.stop:
        stop()
        return 0

    ensure_running()
    if args.sql:
        out = _run(
            "psql", "-h", HOST, "-p", str(PORT), "-U", "postgres", "-d", "postgres",
            "-c", args.sql,
        )
        print((out.stdout or out.stderr).strip())
    elif args.uri_db:
        print(uri(args.uri_db))
    else:
        print(uri("postgres"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
