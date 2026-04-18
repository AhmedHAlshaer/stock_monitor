-- FRED observations (section 5, ENHANCEMENT_GUIDE.md)
CREATE TABLE macro_series (
  series_id text NOT NULL,
  date date NOT NULL,
  value numeric,
  fetched_at timestamptz DEFAULT now(),
  PRIMARY KEY (series_id, date)
);
