"""Minimal test bot - verify handler fires."""
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = "8529551365:AAGbrlhOI17-stm5R71bi_kPggQxCCDTw3o"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[HIT] /start from user_id={update.effective_user.id}", flush=True)
    await update.message.reply_text("✅ Bot is working!")
    print("[HIT] Reply sent!", flush=True)

async def debug_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all for any message."""
    text = update.effective_message.text if update.effective_message else "N/A"
    print(f"[MSG] user={update.effective_user.id} text={text}", flush=True)

async def post_init(app):
    # Send a test message to confirm outbound works
    await app.bot.send_message(chat_id=1486722844, text="🟢 Test bot started — send /start")
    print("[INIT] Test message sent to Henry", flush=True)

def main():
    print("Building app...", flush=True)
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    # Add message handler for debugging
    from telegram.ext import MessageHandler, filters
    app.add_handler(MessageHandler(filters.ALL, debug_all))
    app.post_init = post_init
    print("Starting polling...", flush=True)
    app.run_polling()

if __name__ == "__main__":
    main()
