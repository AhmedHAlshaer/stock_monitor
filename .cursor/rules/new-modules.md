```
When writing NEW modules in core/:
- Start with Pydantic models for all inbound external data.
- Define a clear class (e.g., MacroAnalyzer, SECClient, PositionTracker).
- Expose a narrow public API; make helpers private (_leading_underscore).
- Every class gets a docstring explaining what it does and what it doesn't.
- If the module calls an LLM, the prompt goes in core/prompts/<name>.md, not inline.
- If the module writes to Supabase, it goes through core.persistence.Store,
  never via the supabase client directly.
```
