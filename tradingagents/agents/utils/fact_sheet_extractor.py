"""Fact Sheet Extractor: pulls verified price data from yfinance after analysts complete.

Creates an immutable price reference that gets injected into all downstream agent prompts,
preventing price corruption ($791 → $79) across the multi-agent chain.
"""

import re
import yfinance as yf


def create_fact_sheet_extractor():
    """Create a graph node that extracts a verified price fact sheet."""

    def fact_sheet_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        try:
            stock = yf.Ticker(ticker)

            # Get current/fast info
            fast_info = stock.fast_info
            current_price = getattr(fast_info, "last_price", None)
            fifty_day_avg = getattr(fast_info, "fifty_day_average", None)
            two_hundred_day_avg = getattr(fast_info, "two_hundred_day_average", None)
            year_high = getattr(fast_info, "year_high", None)
            year_low = getattr(fast_info, "year_low", None)

            # Get recent history for support/resistance levels
            hist = stock.history(period="3mo")
            recent_low = None
            recent_high = None
            atr_approx = None
            if not hist.empty:
                # Last 30 days
                recent = hist.tail(30)
                recent_low = float(recent["Low"].min())
                recent_high = float(recent["High"].max())

                # ATR approximation (14-day)
                if len(hist) >= 14:
                    last14 = hist.tail(14)
                    high_low = last14["High"] - last14["Low"]
                    high_close = abs(last14["High"] - last14["Close"].shift(1))
                    low_close = abs(last14["Low"] - last14["Close"].shift(1))
                    true_range = high_low.to_frame("hl")
                    true_range["hc"] = high_close
                    true_range["lc"] = low_close
                    tr_max = true_range.max(axis=1).dropna()
                    if not tr_max.empty:
                        atr_approx = float(tr_max.mean())

            # Determine min/max reasonable price range (50% to 200% of current)
            if current_price:
                min_reasonable = current_price * 0.50
                max_reasonable = current_price * 2.00
            else:
                min_reasonable = None
                max_reasonable = None

            # Build the fact sheet
            lines = [
                "═══════════════════════════════════════════════════════════════",
                "  IMMUTABLE PRICE FACT SHEET — DO NOT MODIFY THESE NUMBERS",
                f"  Ticker: {ticker} | Analysis Date: {trade_date}",
                "═══════════════════════════════════════════════════════════════",
            ]

            if current_price is not None:
                lines.append(f"  Current Price:        ${current_price:,.2f}")
            if year_low is not None:
                lines.append(f"  52-Week Low:          ${year_low:,.2f}")
            if year_high is not None:
                lines.append(f"  52-Week High:         ${year_high:,.2f}")
            if fifty_day_avg is not None:
                lines.append(f"  50-Day SMA:           ${fifty_day_avg:,.2f}")
            if two_hundred_day_avg is not None:
                lines.append(f"  200-Day SMA:          ${two_hundred_day_avg:,.2f}")
            if recent_low is not None:
                lines.append(f"  30-Day Low:           ${recent_low:,.2f}")
            if recent_high is not None:
                lines.append(f"  30-Day High:          ${recent_high:,.2f}")
            if atr_approx is not None:
                lines.append(f"  ATR (14-day approx):  ${atr_approx:,.2f}")

            lines.append("")

            if min_reasonable is not None:
                lines.append(f"  PRICE SANITY CHECK:")
                lines.append(f"  Any price you mention (stop, target, support, resistance)")
                lines.append(f"  MUST be between ${min_reasonable:,.2f} and ${max_reasonable:,.2f}")
                lines.append(f"  (50% to 200% of current price ${current_price:,.2f})")
                lines.append(f"  If you want to reference a price outside this range,")
                lines.append(f"  you MUST explicitly justify why (e.g., 'historical support from 2024').")

            lines.append("")
            lines.append("  SUPPORT LEVELS (nearest):")
            if recent_low is not None:
                lines.append(f"    30-day low:         ${recent_low:,.2f}")
            if fifty_day_avg is not None:
                lines.append(f"    50-day SMA:         ${fifty_day_avg:,.2f}")

            lines.append("")
            lines.append("  RESISTANCE LEVELS (nearest):")
            if recent_high is not None:
                lines.append(f"    30-day high:        ${recent_high:,.2f}")
            if year_high is not None:
                lines.append(f"    52-week high:       ${year_high:,.2f}")

            lines.append("")
            lines.append("  RULE: When setting stops/targets, use these verified numbers.")
            lines.append("  NEVER invent or guess price levels. Reference this sheet.")
            lines.append("═══════════════════════════════════════════════════════════════")

            fact_sheet = "\n".join(lines)

        except Exception as e:
            fact_sheet = f"[Price Fact Sheet unavailable: {e}]"

        return {"price_fact_sheet": fact_sheet}

    return fact_sheet_node
