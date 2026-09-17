"""Slash commands: manual lookups (/mygo) and per-guild setup (/mygoconfig)."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from ..engine import EngineError
from ..render import build_candidates_embed, build_reply
from ..settings import VALID_STYLES

log = logging.getLogger(__name__)


class MyGo(commands.GroupCog, name="mygo"):
    """Ask the model directly, without waiting for it to butt in."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        super().__init__()

    @app_commands.command(name="chat", description="找一張最適合這句話的 MyGO 圖")
    @app_commands.describe(
        text="要餵給模型的文字（繁體中文效果最好）",
        private="只有你看得到結果",
    )
    async def chat(
        self, interaction: discord.Interaction, text: str, private: bool = False
    ) -> None:
        await interaction.response.defer(ephemeral=private, thinking=True)
        try:
            candidates = await self.bot.engine.predict(text, k=1)
        except EngineError as exc:
            await interaction.followup.send(f"模型出錯了：{exc}", ephemeral=True)
            return

        if not candidates:
            await interaction.followup.send("模型沒有給出任何結果。", ephemeral=True)
            return

        settings = self.bot.settings.get(interaction.guild_id)
        # defer(ephemeral=...) only covers the placeholder; a followup is a new
        # message and is public unless it says otherwise.
        await interaction.followup.send(
            **build_reply(candidates[0], settings.style), ephemeral=private
        )

    @app_commands.command(name="candidates", description="列出模型的前幾名候選答案")
    @app_commands.describe(text="要餵給模型的文字", count="要列出幾個（1-10）")
    async def candidates(
        self,
        interaction: discord.Interaction,
        text: str,
        count: app_commands.Range[int, 1, 10] = 5,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            results = await self.bot.engine.predict(text, k=count)
        except EngineError as exc:
            await interaction.followup.send(f"模型出錯了：{exc}", ephemeral=True)
            return

        if not results:
            await interaction.followup.send("模型沒有給出任何結果。", ephemeral=True)
            return

        await interaction.followup.send(embed=build_candidates_embed(results, text))

    @app_commands.command(name="status", description="顯示模型與本伺服器的設定狀態")
    async def status(self, interaction: discord.Interaction) -> None:
        settings = self.bot.settings.get(interaction.guild_id)
        embed = discord.Embed(title="MyGO bot status", color=discord.Color.blurple())
        embed.add_field(name="Engine", value=self.bot.engine.name, inline=True)
        embed.add_field(name="Latency", value=f"{self.bot.latency * 1000:.0f} ms", inline=True)
        embed.add_field(name="Auto-reply", value="on" if settings.enabled else "off", inline=True)
        embed.add_field(name="Chance", value=f"{settings.reply_chance:g}%", inline=True)
        embed.add_field(name="Cooldown", value=f"{settings.cooldown_seconds}s", inline=True)
        embed.add_field(name="Min confidence", value=f"{settings.min_confidence:g}%", inline=True)
        await interaction.response.send_message(embed=embed, ephemeral=True)


def _mention_list(ids: list[int], kind: str) -> str:
    if not ids:
        return "*(none)*"
    prefix = "<#" if kind == "channel" else "<@"
    return " ".join(f"{prefix}{i}>" for i in ids[:20])


@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
class MyGoConfig(commands.GroupCog, name="mygoconfig"):
    """Per-guild tuning. Restricted to members who can manage the server."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        super().__init__()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Enforce Manage Server at runtime.

        `default_permissions` only seeds the default: a server admin can hand
        these commands to any role from Server Settings -> Integrations, so it
        is a UI hint rather than a guarantee. Bot owners listed in OWNER_IDS
        bypass the check so they can fix a guild that locked itself out.
        """
        if interaction.guild is None:
            raise app_commands.NoPrivateMessage()
        if interaction.user.id in self.bot.config.owner_ids:
            return True
        permissions = getattr(interaction.user, "guild_permissions", None)
        if permissions is not None and permissions.manage_guild:
            return True
        raise app_commands.MissingPermissions(["manage_guild"])

    async def _apply(self, interaction: discord.Interaction, **changes) -> None:
        await self._apply_mutation(interaction, lambda _settings: changes, changes.keys())

    async def _apply_mutation(
        self, interaction: discord.Interaction, mutator, changed_keys
    ) -> None:
        """Run the edit inside the store lock, then report what landed."""
        assert interaction.guild_id is not None
        settings = await self.bot.settings.mutate(interaction.guild_id, mutator)
        summary = ", ".join(f"`{k}` → `{getattr(settings, k)}`" for k in changed_keys)
        await interaction.response.send_message(f"已更新：{summary}", ephemeral=True)

    @app_commands.command(name="show", description="顯示本伺服器的完整設定")
    async def show(self, interaction: discord.Interaction) -> None:
        settings = self.bot.settings.get(interaction.guild_id)
        embed = discord.Embed(title="MyGO auto-reply settings", color=discord.Color.blurple())
        embed.add_field(name="enabled", value=str(settings.enabled), inline=True)
        embed.add_field(name="reply_to_mentions", value=str(settings.reply_to_mentions), inline=True)
        embed.add_field(name="reply_chance", value=f"{settings.reply_chance:g}%", inline=True)
        embed.add_field(name="cooldown_seconds", value=str(settings.cooldown_seconds), inline=True)
        embed.add_field(name="min_confidence", value=f"{settings.min_confidence:g}%", inline=True)
        embed.add_field(name="style", value=settings.style, inline=True)
        embed.add_field(
            name="channels (allowlist)",
            value=_mention_list(settings.channels, "channel") if settings.channels else "*(all)*",
            inline=False,
        )
        embed.add_field(
            name="ignored channels",
            value=_mention_list(settings.ignored_channels, "channel"),
            inline=False,
        )
        embed.add_field(
            name="ignored users", value=_mention_list(settings.ignored_users, "user"), inline=False
        )
        embed.add_field(
            name="keywords",
            value=", ".join(f"`{k}`" for k in settings.keywords) or "*(none)*",
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="enable", description="開啟或關閉自動回覆")
    async def enable(self, interaction: discord.Interaction, value: bool) -> None:
        await self._apply(interaction, enabled=value)

    @app_commands.command(name="mentions", description="被 @ 或被回覆時是否一定要回")
    async def mentions(self, interaction: discord.Interaction, value: bool) -> None:
        await self._apply(interaction, reply_to_mentions=value)

    @app_commands.command(name="chance", description="沒被 @ 的訊息要插嘴的機率（0-100）")
    async def chance(
        self, interaction: discord.Interaction, percent: app_commands.Range[float, 0.0, 100.0]
    ) -> None:
        await self._apply(interaction, reply_chance=percent)

    @app_commands.command(name="cooldown", description="同一頻道兩次回覆之間的秒數")
    async def cooldown(
        self, interaction: discord.Interaction, seconds: app_commands.Range[int, 0, 3600]
    ) -> None:
        await self._apply(interaction, cooldown_seconds=seconds)

    @app_commands.command(name="confidence", description="低於這個信心度就不插嘴（0-100）")
    async def confidence(
        self, interaction: discord.Interaction, percent: app_commands.Range[float, 0.0, 100.0]
    ) -> None:
        await self._apply(interaction, min_confidence=percent)

    @app_commands.command(name="style", description="回覆的呈現方式")
    @app_commands.choices(
        value=[app_commands.Choice(name=style, value=style) for style in VALID_STYLES]
    )
    async def style(
        self, interaction: discord.Interaction, value: app_commands.Choice[str]
    ) -> None:
        await self._apply(interaction, style=value.value)

    @app_commands.command(name="channel", description="管理頻道白名單／黑名單")
    @app_commands.choices(
        action=[
            app_commands.Choice(name="allow (add to allowlist)", value="allow"),
            app_commands.Choice(name="unallow (remove from allowlist)", value="unallow"),
            app_commands.Choice(name="ignore (add to blocklist)", value="ignore"),
            app_commands.Choice(name="unignore (remove from blocklist)", value="unignore"),
            app_commands.Choice(name="clear (empty both lists)", value="clear"),
        ]
    )
    async def channel(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        channel: discord.TextChannel | None = None,
    ) -> None:
        if action.value == "clear":
            await self._apply(interaction, channels=[], ignored_channels=[])
            return

        if channel is None:
            await interaction.response.send_message(
                "這個動作需要指定一個頻道。", ephemeral=True
            )
            return

        target = channel.id
        verb = action.value

        def mutator(settings):
            allowed = list(settings.channels)
            ignored = list(settings.ignored_channels)

            if verb == "allow":
                if target not in allowed:
                    allowed.append(target)
                ignored = [c for c in ignored if c != target]
            elif verb == "unallow":
                allowed = [c for c in allowed if c != target]
            elif verb == "ignore":
                if target not in ignored:
                    ignored.append(target)
                allowed = [c for c in allowed if c != target]
            else:  # unignore
                ignored = [c for c in ignored if c != target]

            return {"channels": allowed, "ignored_channels": ignored}

        await self._apply_mutation(
            interaction, mutator, ("channels", "ignored_channels")
        )

    @app_commands.command(name="ignoreuser", description="忽略或恢復某位使用者")
    @app_commands.choices(
        action=[
            app_commands.Choice(name="add", value="add"),
            app_commands.Choice(name="remove", value="remove"),
        ]
    )
    async def ignoreuser(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        user: discord.User,
    ) -> None:
        target = user.id
        adding = action.value == "add"

        def mutator(settings):
            ignored = list(settings.ignored_users)
            if adding:
                if target not in ignored:
                    ignored.append(target)
            else:
                ignored = [u for u in ignored if u != target]
            return {"ignored_users": ignored}

        await self._apply_mutation(interaction, mutator, ("ignored_users",))

    @app_commands.command(name="keyword", description="管理一定會觸發回覆的關鍵字")
    @app_commands.choices(
        action=[
            app_commands.Choice(name="add", value="add"),
            app_commands.Choice(name="remove", value="remove"),
            app_commands.Choice(name="clear", value="clear"),
        ]
    )
    async def keyword(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        word: str | None = None,
    ) -> None:
        if action.value == "clear":
            await self._apply(interaction, keywords=[])
            return

        if not word or not word.strip():
            await interaction.response.send_message(
                "這個動作需要指定一個關鍵字。", ephemeral=True
            )
            return

        normalized = word.strip().lower()
        adding = action.value == "add"

        def mutator(settings):
            keywords = list(settings.keywords)
            if adding:
                if normalized not in keywords:
                    keywords.append(normalized)
            else:
                keywords = [k for k in keywords if k != normalized]
            return {"keywords": keywords}

        await self._apply_mutation(interaction, mutator, ("keywords",))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MyGo(bot))
    await bot.add_cog(MyGoConfig(bot))
