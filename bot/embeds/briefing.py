"""Morning briefing Discord embed — informational only, no buy/sell language."""

from __future__ import annotations

from datetime import datetime, timezone

import discord

from core.briefing import BriefingReport


def _clip(text: str, n: int = 1024) -> str:
    t = text.strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def build_briefing_embed(report: BriefingReport) -> discord.Embed:
    title = f"📊 Morning Briefing — {report.date.strftime('%a, %b %d %Y')}"
    embed = discord.Embed(
        title=title,
        description="Informational snapshot — not investment advice.",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc),
    )

    if report.macro_changes:
        macro_lines = []
        for c in report.macro_changes[:10]:
            macro_lines.append(f"• {c.get('change_str', c.get('series_id', ''))}")
        macro_block = "\n".join(macro_lines) if macro_lines else "—"
    else:
        macro_block = "—"

    embed.add_field(name="🌍 Macro", value=_clip(macro_block), inline=False)

    if report.new_filings:
        flines = []
        for f in report.new_filings[:8]:
            sm = f.get("summary") or "—"
            flines.append(
                f"• **{f.get('symbol')}** {f.get('form_type')} — {sm}\n"
                f"  [link]({f.get('url', '')})"
            )
        filings_block = "\n".join(flines)
    else:
        filings_block = "—"
    embed.add_field(name="📁 Filings (recent)", value=_clip(filings_block), inline=False)

    if report.news_items:
        news_block = "\n".join(f"• {n}" for n in report.news_items[:10])
    else:
        news_block = "—"
    embed.add_field(name="📰 Headlines", value=_clip(news_block), inline=False)

    if report.price_moves:
        px_block = "\n".join(f"• {p}" for p in report.price_moves[:12])
    else:
        px_block = "—"
    embed.add_field(name="📈 Session moves (watchlist)", value=_clip(px_block), inline=False)

    pos = report.position_pnl
    if pos and pos.lines:
        plines = []
        for ln in pos.lines[:8]:
            dd = ""
            if ln.day_change_dollar is not None:
                dd = f" | day ${float(ln.day_change_dollar):+.2f}"
            plines.append(
                f"**{ln.symbol}** {ln.quantity} sh @ ${ln.cost_basis_per_share} → "
                f"${ln.current_price:.2f} | {float(ln.unrealized_pct):+.1f}%{dd}"
            )
        tot = (
            f"Portfolio ~${float(pos.total_market_value):,.0f} | "
            f"unreal ${float(pos.total_unrealized_dollar):+.2f} "
            f"({float(pos.total_unrealized_pct):+.2f}%)"
        )
        if pos.total_day_change_dollar is not None:
            tot += f" | day ${float(pos.total_day_change_dollar):+.2f}"
        pos_block = tot + "\n" + "\n".join(plines)
    else:
        pos_block = "No open positions or prices unavailable."
    embed.add_field(name="💼 Positions", value=_clip(pos_block), inline=False)

    if report.upcoming_earnings:
        earn_block = "\n".join(f"• {e}" for e in report.upcoming_earnings[:8])
    else:
        earn_block = "—"
    embed.add_field(name="⏰ Earnings (upcoming)", value=_clip(earn_block), inline=False)

    syn = getattr(report, "synthesis", None)
    if syn is not None and getattr(syn, "paragraph", None):
        syn_block = (syn.paragraph or "").strip()
        kps = getattr(syn, "key_points", None) or []
        if kps:
            syn_block += "\n\n" + "\n".join(f"• {p}" for p in kps[:3])
        embed.add_field(name="🧠 SYNTHESIS", value=_clip(syn_block), inline=False)

    if report.section_errors:
        err_block = "\n".join(f"• `{e[:200]}`" for e in report.section_errors[:5])
        embed.add_field(
            name="⚠️ Partial data",
            value=_clip(err_block, 1000),
            inline=False,
        )

    embed.set_footer(text="Morning briefing • Informational only")
    return embed
