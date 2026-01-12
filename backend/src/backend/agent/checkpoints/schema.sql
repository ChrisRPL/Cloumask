-- Checkpoint persistence schema for Cloumask agent.
-- This schema extends LangGraph's checkpoint storage with thread management.

-- Thread metadata table
-- Tracks all execution threads and their status
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
    title TEXT,
    input_path TEXT,
    metadata TEXT  -- JSON blob for additional metadata
);

CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at);

-- Checkpoint snapshots table
-- Stores serialized state snapshots for resume capability
CREATE TABLE IF NOT EXISTS checkpoint_snapshots (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    checkpoint_data BLOB NOT NULL,  -- Pickled checkpoint state
    metadata TEXT,  -- JSON blob for checkpoint metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(thread_id, checkpoint_id),
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_thread ON checkpoint_snapshots(thread_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_created ON checkpoint_snapshots(created_at);

-- Pending writes table
-- Stores pending channel writes for LangGraph
CREATE TABLE IF NOT EXISTS writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    value BLOB NOT NULL,  -- Pickled channel value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_writes_thread_checkpoint ON writes(thread_id, checkpoint_id);
