"""SQLAlchemy ORM tables — job sessions and downtime events."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean

from app.db.database import Base


class JobSessionRow(Base):
    """
    One job run on one machine. The per-job counter starts at ~0 when the job
    card is placed and ends at the job's total when the card is removed/changed,
    so production = end_counter - start_counter (≈ end_counter).
    """
    __tablename__ = "job_sessions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    machine_id    = Column(String(64), index=True)
    camera_id     = Column(String(64), index=True)
    job_card      = Column(String(64), index=True)
    started_at    = Column(DateTime, index=True)
    ended_at      = Column(DateTime, nullable=True)
    start_counter = Column(Integer, nullable=True)
    end_counter   = Column(Integer, nullable=True)
    production    = Column(Integer, nullable=True)
    status        = Column(String(16), default="active")  # active | completed


class DowntimeEventRow(Base):
    """A machine-idle period detected from the counter stream."""
    __tablename__ = "downtime_events"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    machine_id    = Column(String(64), index=True)
    camera_id     = Column(String(64), index=True)
    started_at    = Column(DateTime, index=True)
    ended_at      = Column(DateTime, nullable=True)
    duration_s    = Column(Float, nullable=True)
    status        = Column(String(16), default="active")  # active | resolved
    reason        = Column(String(32), nullable=True)
    note          = Column(String(255), default="")
    job_card      = Column(String(64), nullable=True)
    counter_value = Column(Integer, nullable=True)
    time_source   = Column(String(16), default="auto")
