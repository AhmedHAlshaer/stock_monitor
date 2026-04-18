-- User holdings (section 5, ENHANCEMENT_GUIDE.md)
CREATE TABLE positions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol text NOT NULL,
  quantity numeric NOT NULL,
  cost_basis numeric NOT NULL,
  opened_at timestamptz NOT NULL,
  closed_at timestamptz,
  notes text
);

CREATE INDEX idx_positions_symbol_open ON positions(symbol) WHERE closed_at IS NULL;
