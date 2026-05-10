CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id),
    agent_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    latency_ms INTEGER,
    token_count INTEGER,
    policy_violation BOOLEAN DEFAULT FALSE,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id),
    agent_id VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    input JSONB NOT NULL,
    output JSONB,
    latency_ms INTEGER,
    accepted BOOLEAN,
    retry_number INTEGER NOT NULL DEFAULT 0,
    failure_mode VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    triggered_by VARCHAR(50) NOT NULL DEFAULT 'manual',
    prompt_rewrite_id UUID,
    total_cases INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    scores JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    eval_run_id UUID NOT NULL REFERENCES eval_runs(id),
    case_id VARCHAR(50) NOT NULL,
    category VARCHAR(20) NOT NULL,
    query TEXT NOT NULL,
    expected_answer TEXT,
    actual_answer TEXT,
    score_correctness FLOAT,
    score_citation FLOAT,
    score_contradiction FLOAT,
    score_tool_efficiency FLOAT,
    score_budget_compliance FLOAT,
    score_critique_agreement FLOAT,
    justifications JSONB,
    passed BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prompt_rewrites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    eval_run_id UUID NOT NULL REFERENCES eval_runs(id),
    agent_id VARCHAR(50) NOT NULL,
    dimension VARCHAR(50) NOT NULL,
    original_prompt TEXT NOT NULL,
    proposed_prompt TEXT NOT NULL,
    diff TEXT NOT NULL,
    justification TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_logs_job_id ON agent_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_job_id ON tool_calls(job_id);
CREATE INDEX IF NOT EXISTS idx_eval_cases_eval_run_id ON eval_cases(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status ON prompt_rewrites(status);