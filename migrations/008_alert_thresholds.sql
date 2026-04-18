-- Per-ticker % thresholds for intraday / premarket event alerts (Phase 6)
ALTER TABLE watchlist_meta
  ADD COLUMN IF NOT EXISTS alert_threshold_pct numeric;

COMMENT ON COLUMN watchlist_meta.alert_threshold_pct IS 'Abs move % vs ref price to fire price alerts; NULL = use code default';
