-- SEC filings cache + LLM summaries (section 5, ENHANCEMENT_GUIDE.md)
CREATE TABLE filings (
  accession_number text PRIMARY KEY,
  symbol text NOT NULL,
  cik text NOT NULL,
  form_type text NOT NULL,
  filed_at timestamptz NOT NULL,
  period_of_report date,
  url text NOT NULL,
  raw_text text,
  summary text,
  summary_model text,
  summary_at timestamptz,
  alert_sent boolean DEFAULT false
);

CREATE INDEX idx_filings_symbol_filed_at ON filings(symbol, filed_at DESC);
CREATE INDEX idx_filings_form_filed_pending ON filings(form_type, filed_at DESC) WHERE alert_sent = false;
