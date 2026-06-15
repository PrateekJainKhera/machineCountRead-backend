"""
Database layer — SQLAlchemy, company-standard MS SQL Server target.

Config is read from backend/.env (loaded here — no python-dotenv needed).

Two ways to point at MS SQL Server (component style is preferred — no URL
encoding needed for passwords with special characters):

  Component style:
      MCR_DB_SERVER=DESKTOP-I7M84DO          (or SERVER\\INSTANCE)
      MCR_DB_NAME=machinecountread
      MCR_DB_USER=Indus
      MCR_DB_PASSWORD=yourpassword
      MCR_DB_DRIVER=ODBC Driver 17 for SQL Server   (optional)

  Or a full URL:
      MCR_DATABASE_URL=mssql+pyodbc://...

If neither is set → local SQLite file `machinecountread.db` (zero-config).

Persistence is best-effort: if the DB is unreachable, the app still runs fully
in-memory and logs a warning. Production data is never blocked by a DB hiccup.
"""

import os
import logging
import urllib.parse
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

_DEFAULT_SQLITE = "sqlite:///machinecountread.db"


def _load_env_file() -> None:
    """Minimal .env loader (KEY=VALUE lines) — avoids a python-dotenv dependency."""
    env_path = Path(__file__).resolve().parents[2] / ".env"   # backend/.env
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # Real .env values win over whatever was already in the environment
        if key:
            os.environ[key] = val


def _resolve_url():
    """Return (sqlalchemy_url, is_sqlite, human_label)."""
    full = os.environ.get("MCR_DATABASE_URL", "").strip()
    # Ignore the placeholder shipped in the template
    if full and "SERVER/DBNAME" not in full and full != "":
        return full, full.startswith("sqlite"), full.split("@")[-1]

    server = os.environ.get("MCR_DB_SERVER", "").strip()
    name = os.environ.get("MCR_DB_NAME", "").strip()
    if server and name:
        driver = os.environ.get("MCR_DB_DRIVER", "ODBC Driver 17 for SQL Server").strip()
        user = os.environ.get("MCR_DB_USER", "").strip()
        pwd = os.environ.get("MCR_DB_PASSWORD", "")
        # Raw ODBC string passed via odbc_connect — no URL-encoding traps.
        # Encrypt=yes + TrustServerCertificate=yes matches SSMS
        # "Encryption: Mandatory + Trust server certificate".
        parts = [
            f"DRIVER={{{driver}}}",
            f"SERVER={server}",
            f"DATABASE={name}",
            "Encrypt=yes",
            "TrustServerCertificate=yes",
        ]
        if user:
            parts += [f"UID={user}", f"PWD={pwd}"]
        else:
            parts.append("Trusted_Connection=yes")   # Windows auth fallback
        odbc = ";".join(parts)
        url = URL.create("mssql+pyodbc", query={"odbc_connect": odbc})
        return url, False, f"{server}/{name}"

    return _DEFAULT_SQLITE, True, "machinecountread.db (SQLite)"


_engine = None
_SessionLocal = None
_enabled = False


def init_db() -> bool:
    """Create the engine and tables. Returns True if persistence is active."""
    global _engine, _SessionLocal, _enabled
    _load_env_file()
    from app.db import models  # noqa: F401  (registers tables on Base)
    url, is_sqlite, label = _resolve_url()
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    try:
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
        _enabled = True
        logger.info(f"Persistence ENABLED ({'SQLite' if is_sqlite else 'MS SQL Server'}): {label}")
    except Exception as e:
        _enabled = False
        logger.warning(f"Persistence DISABLED — DB init failed ({e}). Running in-memory only.")
    return _enabled


def is_enabled() -> bool:
    return _enabled


@contextmanager
def session_scope():
    """Transactional session. Yields None (and skips) if persistence is off."""
    if not _enabled or _SessionLocal is None:
        yield None
        return
    s = _SessionLocal()
    try:
        yield s
        s.commit()
    except Exception as e:
        s.rollback()
        logger.error(f"DB write failed, rolled back: {e}")
    finally:
        s.close()
