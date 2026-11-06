-- The viewer's tables (Step 6): reviews, the audit log and uploads. For education only, not for
-- clinical use. Reviewer and uploader names are typed in and not verified (there is no login).
--
-- reviews and audit_log are append-only: triggers refuse UPDATE, DELETE and TRUNCATE. The
-- current review state of a study is its newest reviews row. Every insert into
-- inference_results (by the worker) adds an inference_done or inference_failed audit row in the
-- same transaction. inference_results itself is unchanged.

CREATE TABLE IF NOT EXISTS reviews (
    id BIGSERIAL PRIMARY KEY,
    study_uid TEXT NOT NULL,
    inference_result_id BIGINT NOT NULL REFERENCES inference_results (id),
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    reviewer TEXT NOT NULL CHECK (btrim(reviewer) <> ''),
    comment TEXT,
    -- A new UUID from the browser each time the review panel opens: a double submit is refused.
    submit_token UUID NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT reviews_reject_needs_comment
        CHECK (decision = 'approve' OR coalesce(btrim(comment), '') <> '')
);
CREATE INDEX IF NOT EXISTS reviews_study ON reviews (study_uid, id);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL,
    event TEXT NOT NULL
        CHECK (event IN ('upload', 'inference_done', 'inference_failed', 'review')),
    study_uid TEXT,
    details JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_log_study ON audit_log (study_uid, id);

CREATE TABLE IF NOT EXISTS uploads (
    id BIGSERIAL PRIMARY KEY,
    uploader TEXT NOT NULL CHECK (btrim(uploader) <> ''),
    kind TEXT NOT NULL CHECK (kind IN ('png', 'dicom')),
    filename TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
    sha256 TEXT NOT NULL,
    study_uid TEXT,
    orthanc_study_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION refuse_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% on % refused: the table is append-only', TG_OP, TG_TABLE_NAME;
END
$$;

CREATE OR REPLACE TRIGGER reviews_append_only BEFORE UPDATE OR DELETE ON reviews
    FOR EACH ROW EXECUTE FUNCTION refuse_change();
CREATE OR REPLACE TRIGGER reviews_no_truncate BEFORE TRUNCATE ON reviews
    FOR EACH STATEMENT EXECUTE FUNCTION refuse_change();
CREATE OR REPLACE TRIGGER audit_log_append_only BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION refuse_change();
CREATE OR REPLACE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION refuse_change();

-- A review must name a result of the same study.
CREATE OR REPLACE FUNCTION review_matches_result() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM inference_results
        WHERE id = NEW.inference_result_id AND study_uid = NEW.study_uid
    ) THEN
        RAISE EXCEPTION 'inference result % is not a result of study %',
            NEW.inference_result_id, NEW.study_uid;
    END IF;
    RETURN NEW;
END
$$;
CREATE OR REPLACE TRIGGER reviews_match_result BEFORE INSERT ON reviews
    FOR EACH ROW EXECUTE FUNCTION review_matches_result();

CREATE OR REPLACE FUNCTION audit_inference() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO audit_log (actor, event, study_uid, details) VALUES (
        'worker',
        CASE WHEN NEW.status = 'ok' THEN 'inference_done' ELSE 'inference_failed' END,
        NEW.study_uid,
        jsonb_build_object(
            'inference_result_id', NEW.id,
            'model', NEW.model,
            'modality', NEW.modality,
            'checkpoint_sha256', NEW.checkpoint_sha256,
            'seconds', NEW.seconds
        ) || CASE
            WHEN NEW.status = 'ok' THEN '{}'::jsonb
            ELSE jsonb_build_object('error', NEW.results -> 'error')
        END
    );
    RETURN NULL;
END
$$;
CREATE OR REPLACE TRIGGER inference_results_audit AFTER INSERT ON inference_results
    FOR EACH ROW EXECUTE FUNCTION audit_inference();
