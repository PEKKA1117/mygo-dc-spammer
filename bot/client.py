"""The Discord client: wires config, engine, settings and cogs together."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from .config import Config
from .engine import Engine, EngineError, build_engine
from .responder import ReplyPolicy
from .settings import SettingsStore

log = logging.getLogger(__name__)

COGS = ("bot.cogs.autoreply", "bot.cogs.commands")


def build_intents() -> discord.Intents:
    intents = discord.Intents.default()
    # Privileged: enable "Message Content Intent" in the Developer Portal or the
    # bot only ever sees empty strings and auto-reply silently does nothing.
    intents.message_content = True
    return intents


class MyGoBot(commands.Bot):
    def __init__(self, config: Config, engine: Engine | None = None) -> None:
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=build_intents(),
            help_command=None,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.config = config
        self.engine: Engine = engine or build_engine(config)
        self.settings = SettingsStore(config.settings_path)
        self.policy = ReplyPolicy()

    async def setup_hook(self) -> None:
        self.settings.load()
        self.tree.on_error = self._on_app_command_error

        try:
            await self.engine.load()
        except EngineError as exc:
            # Don't take the gateway connection down: slash commands still work
            # and will report the failure, and `load()` is retried per request.
            log.error("Engine failed to load: %s", exc)

        for cog in COGS:
            await self.load_extension(cog)

        if self.config.dev_guild_id:
            guild = discord.Object(id=self.config.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("Synced %d command(s) to dev guild %s", len(synced), self.config.dev_guild_id)
        else:
            synced = await self.tree.sync()
            log.info("Synced %d global command(s) (may take up to an hour to appear)", len(synced))

    async def _on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        """Turn command failures into a reply instead of a silent 'failed' toast."""
        if isinstance(error, app_commands.MissingPermissions):
            message = "你需要「管理伺服器」權限才能改這個設定。"
        elif isinstance(error, app_commands.CommandOnCooldown):
            message = f"太快了，{error.retry_after:.0f} 秒後再試。"
        else:
            log.exception("Unhandled app command error", exc_info=error)
            message = "指令執行失敗，請看 bot log。"

        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except discord.HTTPException:
            pass

    async def on_ready(self) -> None:
        user = self.user
        log.info(
            "Logged in as %s (id=%s) in %d guild(s), engine=%s",
            user,
            getattr(user, "id", "?"),
            len(self.guilds),
            self.engine.name,
        )

    async def close(self) -> None:
        try:
            await self.settings.save()
            await self.engine.close()
        finally:
            await super().close()


__all__ = ["MyGoBot", "build_intents"]
