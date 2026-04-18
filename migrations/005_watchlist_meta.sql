-- Ticker → CIK cache (section 5, ENHANCEMENT_GUIDE.md)
CREATE TABLE watchlist_meta (
  symbol text PRIMARY KEY,
  cik text,
  company_name text,
  bucket text,
  updated_at timestamptz DEFAULT now()
);
