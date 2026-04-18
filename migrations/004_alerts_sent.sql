-- Alert deduplication (section 5, ENHANCEMENT_GUIDE.md)
CREATE TABLE alerts_sent (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  alert_type text NOT NULL,
  reference_id text NOT NULL,
  sent_at timestamptz DEFAULT now(),
  channel_id text,
  payload jsonb
);

CREATE UNIQUE INDEX idx_alerts_sent_type_ref ON alerts_sent(alert_type, reference_id);
