-- The worker's tables as created by src/service/store.py before migrations existed (PR #14).
-- IF NOT EXISTS: a no-op on a database the worker already created. Never edit an applied
-- migration; add a new file instead (src/service/migrate.py checks the sha256).
CREATE TABLE IF NOT EXISTS inference_results (
    id BIGSERIAL PRIMARY KEY,
    study_uid TEXT NOT NULL,
    orthanc_study_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    model TEXT NOT NULL,
    checkpoint_name TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    results JSONB NOT NULL,
    seconds DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (study_uid, checkpoint_sha256)
);
CREATE TABLE IF NOT EXISTS worker_state (
    key TEXT PRIMARY KEY,
    value BIGINT NOT NULL
);
