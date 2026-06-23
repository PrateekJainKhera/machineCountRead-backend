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


class MachineRow(Base):
    """
    Master record for one machine = one camera + its fixed ROIs + settings.

    This is the persistent registry: set up once (camera URL, counter ROI,
    job-card ROI, display type, speeds), and on every backend startup the
    enabled machines are auto-registered into the OCR engine and start reading.
    Survives restarts so 10-50 machines don't need manual re-setup each time.

    `machine_id` doubles as the engine camera_id (one camera per machine).
    """
    __tablename__ = "machines"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    machine_id      = Column(String(64), unique=True, index=True)  # also the camera_id
    source          = Column(String(512))                          # RTSP URL / index / file
    display_type    = Column(String(16), default="lcd")           # lcd | led
    max_rate        = Column(Float, nullable=True)                 # units/sec (jump validation)
    idle_threshold_s = Column(Float, default=300.0)               # counter frozen this long → downtime
    enabled         = Column(Boolean, default=True)               # off = keep config but stop reading

    # Counter ROI (nullable until drawn)
    roi_x = Column(Integer, nullable=True)
    roi_y = Column(Integer, nullable=True)
    roi_w = Column(Integer, nullable=True)
    roi_h = Column(Integer, nullable=True)

    # Job-card slot ROI (nullable until drawn)
    jc_x = Column(Integer, nullable=True)
    jc_y = Column(Integer, nullable=True)
    jc_w = Column(Integer, nullable=True)
    jc_h = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)


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
