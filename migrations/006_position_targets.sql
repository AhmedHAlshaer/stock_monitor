-- Optional profit target / stop-loss thresholds (% vs cost basis) for Phase 6 alerts
ALTER TABLE positions
  ADD COLUMN IF NOT EXISTS target_pct numeric,
  ADD COLUMN IF NOT EXISTS stop_pct numeric;

COMMENT ON COLUMN positions.target_pct IS 'Optional take-profit threshold as percent gain vs cost (e.g. 15 = +15%%)';
COMMENT ON COLUMN positions.stop_pct IS 'Optional stop-loss threshold as percent loss vs cost (e.g. 10 = -10%%)';
