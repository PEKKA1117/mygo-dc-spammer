"""Entry point: `python -m bot`."""

from __future__ import annotations

import logging
import sys

import discord
from dotenv import load_dotenv

from .client import MyGoBot
from .config import Config


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # discord.py's own gateway chatter is noisy at DEBUG.
    logging.getLogger("discord").setLevel(logging.WARNING)


def main() -> int:
    load_dotenv()

    try:
        config = Config.from_env()
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    log = logging.getLogger("bot")

    bot = MyGoBot(config)
    try:
        bot.run(config.token, log_handler=None)
    except discord.LoginFailure:
        log.error("Discord rejected the token. Check DISCORD_TOKEN in your .env.")
        return 1
    except discord.PrivilegedIntentsRequired:
        log.error(
            "The Message Content intent is not enabled. Turn it on at "
            "https://discord.com/developers/applications -> your app -> Bot -> "
            "Privileged Gateway Intents."
        )
        return 1
    except KeyboardInterrupt:
        log.info("Shutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
