"""The auto-reply listener: watches messages and answers with a MyGO image."""

from __future__ import annotations

import logging
import time

import discord
from discord.ext import commands

from ..engine import EngineError
from ..render import build_reply
from ..responder import MessageContext, Trigger, pick_candidate

log = logging.getLogger(__name__)


class AutoReply(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    def _build_context(self, message: discord.Message) -> MessageContext:
        me = self.bot.user
        my_id = getattr(me, "id", 0)
        is_dm = message.guild is None

        mentions_bot = any(user.id == my_id for user in message.mentions)
        if is_dm:
            # In a DM every message is addressed to the bot by definition.
            mentions_bot = True

        return MessageContext(
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
            author_id=message.author.id,
            content=message.content or "",
            author_is_bot=message.author.bot,
            is_self=message.author.id == my_id,
            mentions_bot=mentions_bot,
            is_reply_to_bot=self._is_reply_to_bot(message, my_id),
        )

    @staticmethod
    def _is_reply_to_bot(message: discord.Message, my_id: int) -> bool:
        reference = message.reference
        if reference is None:
            return False
        referenced = reference.resolved or reference.cached_message
        if isinstance(referenced, discord.Message):
            return referenced.author.id == my_id
        return False

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if self.bot.user is None:
            return

        ctx = self._build_context(message)
        settings = self.bot.settings.get(ctx.guild_id)
        now = time.monotonic()

        decision = self.bot.policy.decide(ctx, settings, now)
        if not decision.should_reply or decision.trigger is None:
            if decision.reason not in ("own message", "author is a bot", "no trigger matched"):
                log.debug("Skipping message %s: %s", message.id, decision.reason)
            return

        # Claim the cooldown slot before the slow part, so a burst of messages
        # arriving together cannot each pass the check and all reply. Every path
        # that ends without speaking hands the slot back.
        previous = self.bot.policy.record_reply(ctx.channel_id, now)

        try:
            async with message.channel.typing():
                candidates = await self.bot.engine.predict(
                    decision.text, k=self.bot.config.candidate_count
                )
        except EngineError as exc:
            log.error("Prediction failed for message %s: %s", message.id, exc)
            if decision.trigger is Trigger.MENTION:
                await self._safe_send(message, content="模型現在壞掉了，等等再試 🐧")
            else:
                self.bot.policy.release_reply(ctx.channel_id, previous)
            return
        except discord.HTTPException as exc:
            log.warning("Could not open typing indicator in %s: %s", ctx.channel_id, exc)
            self.bot.policy.release_reply(ctx.channel_id, previous)
            return

        candidate = pick_candidate(candidates, decision.trigger, settings)
        if candidate is None:
            log.debug(
                "No candidate above %.1f%% for message %s",
                settings.min_confidence,
                message.id,
            )
            self.bot.policy.release_reply(ctx.channel_id, previous)
            return

        await self._safe_send(message, **build_reply(candidate, settings.style))

    async def _safe_send(self, message: discord.Message, **kwargs) -> None:
        try:
            await message.reply(mention_author=False, **kwargs)
        except discord.Forbidden:
            log.warning("Missing permission to reply in channel %s", message.channel.id)
        except discord.NotFound:
            # The message was deleted between prediction and reply.
            log.debug("Message %s vanished before the reply landed", message.id)
        except discord.HTTPException as exc:
            log.warning("Failed to reply to message %s: %s", message.id, exc)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AutoReply(bot))
