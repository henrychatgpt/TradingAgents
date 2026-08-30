"""
TradingAgents Telegram Bot
===========================
Multi-user Telegram bot for running TradingAgents stock analysis.

Commands:
  /start   — Register & welcome message
  /analyze <TICKER> — Queue a TradingAgents analysis (one at a time)
  /status  — Show queue position / current run
  /cancel  — Cancel your queued or running analysis
  /done    — End discussion mode
  /judge [TICKER [DATE]] — Evaluate analysis quality (LLM-as-Judge)
  /calibrate [TICKER] [DATE] — Calibrate judge scores vs market outcomes
  /refine TICKER [DATE] — Refine weak analyses using judge feedback
  /help    — Usage info

Concurrency: Only ONE analysis runs at a time (OpenCode Go API rate limit).
             Additional requests are queued and processed FIFO.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# Discussion module (Q&A + mini-debate)
from discussion import (
    DiscussionState,
    discussions,
    is_challenge,
    answer_question,
    run_mini_debate,
)

# Judge module (LLM-as-Judge quality evaluation)
from judge import Judge, JudgeResult

# Calibration module (judge scores vs market outcomes)
from judge_calibration import JudgeCalibration

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path("/opt/data/workspace/TradingAgents")
SECRETS_FILE = Path("/data/.hermes/secrets/tradingagents-bot.env")
RESULTS_DIR = BASE_DIR / "results"
RUNNER_SCRIPT = BASE_DIR / "scripts" / "run_analysis.py"
VENV_PYTHON = str(BASE_DIR / ".venv" / "bin" / "python")

load_dotenv(SECRETS_FILE)
load_dotenv(Path("/opt/data/workspace/TradingAgents/.env"))  # OPENCODE_GO_API_KEY for discussion
BOT_TOKEN = os.getenv("TRADINGAGENTS_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TRADINGAGENTS_BOT_TOKEN not set")
print(f"[CONFIG] Token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}", flush=True)

# Whitelisted user IDs
ALLOWED_USERS = {
    1486722844,  # Henry
    1203722519,  # Friend
}

HKT = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("ta_bot")
# Silence HTTP request noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Queue / State
# ---------------------------------------------------------------------------

@dataclass
class AnalysisJob:
    user_id: int
    chat_id: int
    ticker: str
    date: Optional[str] = None
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"  # queued | running | done | failed | cancelled
    result_file: Optional[str] = None
    error: Optional[str] = None

class AnalysisQueue:
    """FIFO queue with single-worker semantics."""

    def __init__(self):
        self._queue: list[AnalysisJob] = []
        self._current: Optional[AnalysisJob] = None
        self._lock = asyncio.Lock()
        self._worker_running = False
        self._bot = None  # set once at startup

    def set_bot(self, bot):
        self._bot = bot

    @property
    def current(self) -> Optional[AnalysisJob]:
        return self._current

    def queued_jobs(self, user_id: int) -> list[AnalysisJob]:
        return [j for j in self._queue if j.user_id == user_id]

    def position(self, user_id: int) -> Optional[int]:
        for i, j in enumerate(self._queue):
            if j.user_id == user_id:
                return i + 1
        return None

    async def enqueue(self, job: AnalysisJob) -> int:
        async with self._lock:
            # Check duplicate
            if self._current and self._current.user_id == job.user_id and self._current.ticker == job.ticker and self._current.status == "running":
                return -1  # duplicate running
            for j in self._queue:
                if j.user_id == job.user_id and j.ticker == job.ticker:
                    return -1  # duplicate queued
            self._queue.append(job)
            pos = len(self._queue)
        # Start worker if not running
        if not self._worker_running:
            asyncio.create_task(self._worker())
        return pos

    async def cancel_user(self, user_id: int) -> Optional[AnalysisJob]:
        async with self._lock:
            # Check queue first
            for i, j in enumerate(self._queue):
                if j.user_id == user_id:
                    j.status = "cancelled"
                    self._queue.pop(i)
                    return j
            # Check current
            if self._current and self._current.user_id == user_id:
                self._current.status = "cancelled"
                return self._current
        return None

    async def _worker(self):
        if self._worker_running:
            return
        self._worker_running = True
        try:
            while True:
                async with self._lock:
                    if not self._queue:
                        break
                    job = self._queue.pop(0)
                    self._current = job
                await self._run_job(job)
                # Send result directly BEFORE clearing _current
                if self._bot:
                    try:
                        await send_analysis_result(job.chat_id, job, self._bot)
                        print(f"[NOTIFY] Result sent for {job.ticker} to {job.chat_id}", flush=True)
                    except Exception as e:
                        print(f"[NOTIFY] Failed to send result for {job.ticker}: {e}", flush=True)
                    # Set up discussion context
                    try:
                        full_log_dir = RESULTS_DIR / job.ticker.upper() / "TradingAgentsStrategy_logs"
                        if full_log_dir.exists():
                            logs = sorted(full_log_dir.glob("full_states_log_*.json"), reverse=True)
                            if logs:
                                with open(logs[0]) as f:
                                    data = json.load(f)
                                decision = data.get("final_trade_decision", "")
                                if decision:
                                    discussions[job.user_id] = DiscussionState(
                                        ticker=job.ticker,
                                        analysis_context=decision,
                                        full_log_path=str(logs[0]),
                                    )
                                    await self._bot.send_message(
                                        chat_id=job.chat_id,
                                        text=f"💬 Reply to discuss with the analysis team about {job.ticker}!\n"
                                             "Challenge the decision to trigger a mini-debate.\n"
                                             "/done to end discussion."
                                    )
                                    print(f"[DISCUSS] Context set for {job.ticker} user {job.user_id}", flush=True)
                    except Exception as e:
                        print(f"[DISCUSS] Failed to set context: {e}", flush=True)
                async with self._lock:
                    self._current = None
        finally:
            self._worker_running = False

    async def _run_job(self, job: AnalysisJob):
        job.started_at = time.time()
        date_str = job.date or self._auto_date()
        result_file = RESULTS_DIR / f"{job.ticker.lower()}_{date_str}.json"
        full_log = RESULTS_DIR / job.ticker.upper() / "TradingAgentsStrategy_logs" / f"full_states_log_{date_str}.json"

        # --- Cache check: reuse existing result if same ticker + date ---
        if result_file.exists() and full_log.exists():
            job.status = "done"
            job.finished_at = time.time()
            job.result_file = str(result_file)
            log.info(f"Cache hit: {job.ticker} ({date_str}) — reusing existing result")
            print(f"[CACHE] {job.ticker} ({date_str}) — result already exists", flush=True)
            return

        # --- Run fresh analysis ---
        job.status = "running"
        log.info(f"Starting analysis: {job.ticker} for user {job.user_id}")

        # Build command
        cmd = [VENV_PYTHON, str(RUNNER_SCRIPT), "--ticker", job.ticker]
        if job.date:
            cmd.extend(["--date", job.date])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            if proc.returncode == 0:
                job.status = "done"
                job.finished_at = time.time()
                # Find result file
                date_str = job.date or self._auto_date()
                result_file = RESULTS_DIR / f"{job.ticker.lower()}_{date_str}.json"
                if result_file.exists():
                    job.result_file = str(result_file)
                log.info(f"Analysis done: {job.ticker}")
            else:
                job.status = "failed"
                job.error = output[-2000:] if output else "Unknown error"
                log.error(f"Analysis failed: {job.ticker} — rc={proc.returncode}")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.exception(f"Analysis exception: {job.ticker}")

    @staticmethod
    def _auto_date() -> str:
        """Return yesterday's date (last trading day heuristic)."""
        now_hkt = datetime.now(HKT)
        # If weekend, use Friday
        if now_hkt.weekday() == 5:  # Saturday
            now_hkt -= timedelta(days=1)
        elif now_hkt.weekday() == 6:  # Sunday
            now_hkt -= timedelta(days=2)
        elif now_hkt.hour < 5:  # Before 5am HKT, use previous day
            now_hkt -= timedelta(days=1)
            if now_hkt.weekday() == 5:
                now_hkt -= timedelta(days=1)
            elif now_hkt.weekday() == 6:
                now_hkt -= timedelta(days=2)
        return now_hkt.strftime("%Y-%m-%d")


queue = AnalysisQueue()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def authorized(user_id: int) -> bool:
    return user_id in ALLOWED_USERS

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

async def send_analysis_result(chat_id: int, job: AnalysisJob, bot):
    """Send the final analysis result to user."""
    if job.status == "done":
        # Read summary
        if job.result_file and Path(job.result_file).exists():
            with open(job.result_file) as f:
                summary = json.load(f)
            decision = summary.get("decision", "N/A")
            date = summary.get("date", "N/A")
        else:
            decision = "N/A"
            date = "N/A"

        # Try to read full decision from log
        full_log_dir = RESULTS_DIR / job.ticker.upper() / "TradingAgentsStrategy_logs"
        full_decision = None
        if full_log_dir.exists():
            logs = sorted(full_log_dir.glob("full_states_log_*.json"), reverse=True)
            if logs:
                try:
                    with open(logs[0]) as f:
                        data = json.load(f)
                    full_decision = data.get("final_trade_decision", "")
                except Exception:
                    pass

        duration = fmt_duration(job.finished_at - job.started_at) if job.finished_at and job.started_at else "N/A"
        cached = (job.finished_at - job.started_at) < 5 if job.finished_at and job.started_at else False
        cache_tag = " (cached)" if cached else ""

        header = (
            f"📊 Analysis Complete: {job.ticker.upper()}{cache_tag}\n"
            f"Date: {date}\n"
            f"Decision: {decision}\n"
            f"Duration: {duration}\n"
        )

        if full_decision:
            header = header + "\n"
            # Split into chunks to avoid Telegram's 4096-char limit
            MAX_MSG = 4096
            # First send header + initial content
            remaining = full_decision
            is_first = True
            while remaining:
                if is_first:
                    body_space = MAX_MSG - len(header)
                    chunk = header + remaining[:body_space]
                    remaining = remaining[body_space:]
                    is_first = False
                else:
                    chunk = remaining[:MAX_MSG]
                    remaining = remaining[MAX_MSG:]
                await bot.send_message(chat_id=chat_id, text=chunk)
                # Small delay between multi-part messages to avoid rate limits
                if remaining:
                    await asyncio.sleep(0.3)
        else:
            await bot.send_message(chat_id=chat_id, text=header)

    elif job.status == "failed":
        err_msg = job.error or "Unknown error"
        if len(err_msg) > 3000:
            err_msg = err_msg[-3000:]
        await bot.send_message(
            chat_id=chat_id,
            text=f"❌ Analysis Failed: {job.ticker.upper()}\n\n{err_msg}",
        )

    elif job.status == "cancelled":
        await bot.send_message(
            chat_id=chat_id,
            text=f"🚫 Analysis cancelled: {job.ticker.upper()}",
        )

# ---------------------------------------------------------------------------
# Notification task — polls for completed jobs
# ---------------------------------------------------------------------------

async def notification_loop(bot):
    """Background task that sends results when analyses complete."""
    notified_jobs: set = set()
    while True:
        await asyncio.sleep(5)
        # Check current job
        job = queue.current
        if job and job.status in ("done", "failed", "cancelled"):
            job_id = id(job)
            if job_id not in notified_jobs:
                notified_jobs.add(job_id)
                try:
                    await send_analysis_result(job.chat_id, job, bot)
                except Exception as e:
                    log.exception(f"Failed to send result for {job.ticker}")
        # Clean up old notified jobs (keep last 100)
        if len(notified_jobs) > 100:
            notified_jobs.clear()

# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = update.effective_user.first_name or "there"
    print(f"[CMD] /start from user_id={user_id} name={name}", flush=True)

    if not authorized(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you are not authorized to use this bot."
        )
        print(f"[CMD] Unauthorized user {user_id}", flush=True)
        return

    print(f"[CMD] Sending welcome to {user_id}", flush=True)
    try:
        await update.message.reply_text(
            f"👋 Hi {name}! I'm the TradingAgents Analyzer bot.\n\n"
            "I run multi-agent AI stock analyses (~25 min each).\n\n"
            "Commands:\n"
            "/analyze TICKER - Run analysis\n"
            "/status - Check queue/progress\n"
            "/cancel - Cancel your analysis\n"
            "/help - More info\n\n"
            "One analysis runs at a time. Others are queued."
        )
        print(f"[CMD] Welcome sent OK to {user_id}", flush=True)
    except Exception as e:
        print(f"[CMD] FAILED to send welcome: {e}", flush=True)
        # Fallback: send via bot.send_message
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"Hi {name}! Bot is working. Send /help for commands.")
            print(f"[CMD] Fallback send OK", flush=True)
        except Exception as e2:
            print(f"[CMD] Fallback also failed: {e2}", flush=True)

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if not authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    # Clear any existing discussion
    if user_id in discussions:
        old = discussions.pop(user_id)
        print(f"[DISCUSS] Cleared old discussion for {old.ticker}", flush=True)

    if not context.args:
        await update.message.reply_text(
            "⚠️ Please provide a ticker symbol.\n\n"
            "Example: `/analyze AAPL` or `/analyze NVDA 2026-05-01`"
        )
        return

    ticker = context.args[0].upper().strip()
    date = context.args[1] if len(context.args) > 1 else None

    # Validate ticker format (letters, numbers, dots, hyphens — e.g. AAPL, 100.HK, BRK-B)
    import re
    if not re.match(r'^[A-Za-z0-9.\-]{1,15}$', ticker):
        await update.message.reply_text("⚠️ Invalid ticker format. Use letters, numbers, dots, or hyphens, e.g. `AAPL`, `100.HK`.")
        return

    # Validate date format if provided
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            await update.message.reply_text("⚠️ Invalid date format. Use YYYY-MM-DD.")
            return

    job = AnalysisJob(
        user_id=user_id,
        chat_id=chat_id,
        ticker=ticker,
        date=date,
    )

    pos = await queue.enqueue(job)

    if pos == -1:
        await update.message.reply_text(
            f"⚠️ You already have a **{ticker}** analysis queued or running."
        )
        return

    current = queue.current
    if current and current != job:
        wait_msg = f" (queue position: **#{pos}** — waiting for {current.ticker} to finish)"
    else:
        wait_msg = " (starting now ⚡)"

    await update.message.reply_text(
        f"📋 **{ticker}** analysis queued!{wait_msg}\n\n"
        f"Use `/status` to check progress.\n"
        f"Estimated time: ~25 minutes"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not authorized(user_id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    current = queue.current
    queued = queue.queued_jobs(user_id)
    pos = queue.position(user_id)

    lines = ["📊 **Queue Status**\n"]

    if current:
        elapsed = time.time() - current.started_at if current.started_at else 0
        if current.user_id == user_id:
            lines.append(f"🔄 **Your analysis: {current.ticker}** — running ({fmt_duration(elapsed)} elapsed)")
        else:
            lines.append(f"🔄 **Current: {current.ticker}** (another user, {fmt_duration(elapsed)} elapsed)")

    if queued:
        for j in queued:
            lines.append(f"⏳ **Your queued: {j.ticker}** — position #{pos or '?'}")
    elif not current or current.user_id != user_id:
        lines.append("No analyses queued or running for you.")

    await update.message.reply_text("\n".join(lines))

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not authorized(user_id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    cancelled = await queue.cancel_user(user_id)
    if cancelled:
        if cancelled.status == "running":
            await update.message.reply_text(
                f"🚫 Marked **{cancelled.ticker}** for cancellation. "
                "The running process will finish its current step then stop."
            )
        else:
            await update.message.reply_text(f"🚫 Cancelled **{cancelled.ticker}** analysis (was queued).")
    else:
        await update.message.reply_text("ℹ️ No queued or running analysis found for you.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not authorized(user_id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    await update.message.reply_text(
        "TradingAgents Analyzer Bot\n\n"
        "Runs 13-agent AI stock analysis (Market, Social, News, Fundamentals -> "
        "Bull/Bear Debate -> Research Manager -> Trader -> Risk Debate -> Portfolio Manager).\n\n"
        "Commands:\n"
        "/analyze TICKER - Analyze a stock\n"
        "/analyze TICKER YYYY-MM-DD - Analyze with specific date\n"
        "/status - Queue position & progress\n"
        "/cancel - Cancel your analysis\n"
        "/done - End discussion\n"
        "/judge [TICKER [DATE]] - Evaluate analysis quality (LLM-as-Judge)\n"
        "/calibrate [TICKER] [DATE] - Judge scores vs market outcomes\n"
        "/refine TICKER [DATE] - Refine weak analyses using judge feedback\n\n"
        "After analysis, reply to discuss with the team!\n"
        "Challenge the decision to trigger a mini-debate (Bull vs Bear).\n\n"
        "Notes:\n"
        "- One analysis at a time (~25 min each)\n"
        "- If busy, your request is queued\n"
        "- Uses verified price fact sheets to prevent errors"
    )

# ---------------------------------------------------------------------------
# /judge — Evaluate analysis quality
# ---------------------------------------------------------------------------

async def cmd_judge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Judge the quality of a completed analysis. Usage: /judge [TICKER [DATE]]"""
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    args = context.args or []
    ticker = args[0].upper() if args else None
    date = args[1] if len(args) > 1 else None

    # If no ticker, try to use the user's last completed analysis
    if not ticker:
        # Check if there's a discussion with a ticker
        if user_id in discussions:
            ticker = discussions[user_id].ticker
        else:
            await update.message.reply_text(
                "Usage: /judge TICKER [DATE]\n"
                "Example: /judge TSLA 2026-05-01\n"
                "If DATE is omitted, uses the latest available."
            )
            return

    # Find the latest log if no date
    if not date:
        log_dir = RESULTS_DIR / ticker.upper() / "TradingAgentsStrategy_logs"
        if log_dir.exists():
            logs = sorted(log_dir.glob("full_states_log_*.json"), reverse=True)
            if logs:
                date = logs[0].stem.replace("full_states_log_", "")
        if not date:
            await update.message.reply_text(f"No analysis found for {ticker}.")
            return

    # Check if judge already ran for this
    judge = Judge()
    existing = judge.load_result(ticker, date)
    if existing and not existing.error:
        await update.message.reply_text(
            f"Previously evaluated:\n\n{existing.to_telegram()}"
        )
        return

    await update.message.reply_text(
        f"Evaluating {ticker} ({date})...\n"
        f"This takes ~30-60s. I'll send the results when ready."
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        result = await judge.evaluate(ticker, date)
        if result.error:
            await update.message.reply_text(f"Judge error: {result.error}")
            return

        # Save result
        path = judge.save_result(result)

        # Send evaluation
        msg = result.to_telegram()
        if len(msg) > 4096:
            msg = msg[:4090] + "\n..."
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"Judge failed: {e}")

# ---------------------------------------------------------------------------
# /calibrate — Correlate judge scores with market outcomes
# ---------------------------------------------------------------------------

async def cmd_calibrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show calibration report: judge scores vs actual market outcomes."""
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    args = context.args or []
    ticker = args[0].upper() if args else None
    date = args[1] if len(args) > 1 else None

    await update.message.reply_text(
        "Running calibration... Fetching price data and computing outcomes."
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        cal = JudgeCalibration()
        report = cal.run(ticker=ticker, date=date)

        if report.error:
            await update.message.reply_text(f"Calibration error: {report.error}")
            return

        if report.total_analyses == 0:
            await update.message.reply_text(
                "No analyses with judge scores found. Run /judge on some analyses first."
            )
            return

        # Save report
        path = cal.save_report(report)

        # Send report
        msg = report.to_telegram()
        if len(msg) > 4096:
            msg = msg[:4090] + "\n..."
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"Calibration failed: {e}")

# ---------------------------------------------------------------------------
# /refine — Self-critique refinement
# ---------------------------------------------------------------------------

async def cmd_refine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refine analysis using judge feedback. Usage: /refine TICKER [DATE] [--force]"""
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /refine TICKER [DATE]\n"
            "Example: /refine NVDA 2026-05-01\n\n"
            "Refines weak analyses (score < 7.5) using judge feedback.\n"
            "Add --force to override the threshold."
        )
        return

    ticker = args[0].upper()
    date = None
    force = "--force" in args

    for a in args[1:]:
        if a == "--force":
            continue
        if not date:
            date = a

    if not date:
        date = AnalysisQueue._auto_date()

    await update.message.reply_text(
        f"Refining {ticker} ({date})...\n"
        "Step 1: Generating revised decision from judge feedback"
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        from judge_refine import Refiner
        refiner = Refiner()
        result = await refiner.refine(ticker, date, force=force)

        if result.error:
            await update.message.reply_text(f"Refine: {result.error}")
            return

        refiner.save_result(result)
        msg = result.to_telegram()
        if len(msg) > 4096:
            msg = msg[:4090] + "\n..."
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"Refinement failed: {e}")

# ---------------------------------------------------------------------------
# /done — End discussion
# ---------------------------------------------------------------------------

async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return
    if user_id in discussions:
        ticker = discussions[user_id].ticker
        del discussions[user_id]
        await update.message.reply_text(f"Discussion ended for {ticker}. Send /analyze for new analysis.")
    else:
        await update.message.reply_text("No active discussion. Send /analyze TICKER to start.")

# ---------------------------------------------------------------------------
# Discussion Message Handler (replaces debug handler)
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle non-command messages — route to Q&A or mini-debate."""
    user_id = update.effective_user.id
    text = update.effective_message.text if update.effective_message else ""
    
    if not text:
        return

    # No active discussion
    if user_id not in discussions:
        if authorized(user_id):
            await update.message.reply_text(
                "No active analysis to discuss. Send /analyze TICKER first!"
            )
        return

    state = discussions[user_id]

    # Send typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    if is_challenge(text):
        # Mini-debate mode (3 LLM calls)
        await update.message.reply_text(
            "🔄 Running mini-debate on your challenge...\n"
            "Bull vs Bear vs Moderator — ~3-5 min"
        )
        try:
            result = await run_mini_debate(state, text)
            # Split long messages (Telegram 4096 char limit)
            if len(result) > 4096:
                parts = []
                while result:
                    parts.append(result[:4090])
                    result = result[4090:]
                for i, part in enumerate(parts):
                    prefix = f"({i+1}/{len(parts)}) " if len(parts) > 1 else ""
                    await update.message.reply_text(prefix + part)
            else:
                await update.message.reply_text(result)
            state.history.append({"role": "user", "content": text})
            state.history.append({"role": "assistant", "content": result[:1000]})
        except Exception as e:
            await update.message.reply_text(f"Debate error: {e}")
    else:
        # Q&A mode (1 LLM call)
        try:
            result = await answer_question(state, text)
            await update.message.reply_text(result)
            state.history.append({"role": "user", "content": text})
            state.history.append({"role": "assistant", "content": result})
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting TradingAgents Telegram Bot...")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("done", cmd_done))
    app.add_handler(CommandHandler("judge", cmd_judge))
    app.add_handler(CommandHandler("calibrate", cmd_calibrate))
    app.add_handler(CommandHandler("refine", cmd_refine))

    # Discussion handler — all non-command text messages
    from telegram.ext import MessageHandler, filters
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start: set bot reference on queue
    async def post_init(application):
        queue.set_bot(application.bot)
        await application.bot.send_message(
            chat_id=1486722844,
            text="🟢 TRTA Bot restarted — ready to analyze!"
        )
        print("[INIT] Bot ready", flush=True)

    app.post_init = post_init

    # Add error handler to prevent crash on transient errors
    async def error_handler(update, context):
        print(f"[ERROR] {context.error}", flush=True)

    app.add_error_handler(error_handler)

    log.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
