"""Job model and database operations for GPT-OSS SLURM backend"""
import sqlite3
import json
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_PATH


class JobStatus(str, Enum):
    QUEUED = "queued"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Job:
    id: str
    prompt: str
    status: JobStatus
    slurm_job_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    system_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    partition: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value if isinstance(self.status, JobStatus) else self.status
        return d


def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            slurm_job_id TEXT,
            result TEXT,
            error TEXT,
            system_message TEXT,
            partition TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def create_job(prompt: str, system_message: str = None) -> Job:
    """Create a new job and save to database"""
    init_db()
    
    job = Job(
        id=str(uuid.uuid4()),
        prompt=prompt,
        status=JobStatus.QUEUED,
        system_message=system_message,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO jobs (id, prompt, status, system_message, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        job.id,
        job.prompt,
        job.status.value,
        job.system_message,
        job.created_at,
        job.updated_at
    ))
    conn.commit()
    conn.close()
    
    return job


def get_job(job_id: str) -> Optional[Job]:
    """Get a job by ID"""
    init_db()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return Job(
            id=row[0],
            prompt=row[1],
            status=JobStatus(row[2]),
            slurm_job_id=row[3],
            result=row[4],
            error=row[5],
            system_message=row[6],
            partition=row[7],
            created_at=row[8],
            updated_at=row[9]
        )
    return None


def update_job(job_id: str, **kwargs) -> Optional[Job]:
    """Update a job with new values"""
    init_db()
    
    # Convert JobStatus to string if needed
    if 'status' in kwargs and isinstance(kwargs['status'], JobStatus):
        kwargs['status'] = kwargs['status'].value
    
    kwargs['updated_at'] = datetime.utcnow().isoformat()
    
    set_clause = ', '.join([f'{k} = ?' for k in kwargs.keys()])
    values = list(kwargs.values()) + [job_id]
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'UPDATE jobs SET {set_clause} WHERE id = ?', values)
    conn.commit()
    conn.close()
    
    return get_job(job_id)


def get_all_jobs(limit: int = 100) -> List[Job]:
    """Get all jobs ordered by creation time"""
    init_db()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    jobs = []
    for row in rows:
        jobs.append(Job(
            id=row[0],
            prompt=row[1],
            status=JobStatus(row[2]),
            slurm_job_id=row[3],
            result=row[4],
            error=row[5],
            system_message=row[6],
            partition=row[7],
            created_at=row[8],
            updated_at=row[9]
        ))
    return jobs


def delete_job(job_id: str) -> bool:
    """Delete a job by ID"""
    init_db()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted
