"""Smoke tests that exercise the discord.py-facing layer without a gateway."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

from bot.client import MyGoBot, build_intents
from bot.cogs.autoreply import AutoReply
from bot.config import Config
from bot.engine import Candidate, RandomEngine
from bot.render import build_candidates_embed, build_reply


def make_config(tmp_path: Path) -> Config:
    return Config(
        token="not-a-real-token",
        engine="random",
        mygochat_path=tmp_path / "vendor",
        settings_path=tmp_path / "guilds.json",
        max_input_chars=300,
        candidate_count=5,
    )


@pytest.fixture
async def bot(tmp_path):
    instance = MyGoBot(make_config(tmp_path), engine=RandomEngine(seed=1))
    yield instance
    await instance.http.close()


class TestIntents:
    def test_message_content_is_requested(self):
        # Without this the auto-reply listener only ever sees empty strings.
        assert build_intents().message_content is True


class TestExtensionLoading:
    @pytest.mark.asyncio
    async def test_all_cogs_load(self, bot):
        for extension in ("bot.cogs.autoreply", "bot.cogs.commands"):
            await bot.load_extension(extension)
        # GroupCogs register under their group name, not the class name.
        assert set(bot.cogs) >= {"AutoReply", "mygo", "mygoconfig"}

    @pytest.mark.asyncio
    async def test_slash_commands_are_registered(self, bot):
        await bot.load_extension("bot.cogs.commands")
        groups = {command.name for command in bot.tree.get_commands()}
        assert {"mygo", "mygoconfig"} <= groups

        mygo = discord.utils.get(bot.tree.get_commands(), name="mygo")
        assert {c.name for c in mygo.commands} == {"chat", "candidates", "status"}

    @pytest.mark.asyncio
    async def test_config_group_requires_manage_guild(self, bot):
        await bot.load_extension("bot.cogs.commands")
        config_group = discord.utils.get(bot.tree.get_commands(), name="mygoconfig")
        assert config_group.default_permissions.manage_guild is True
        assert config_group.guild_only is True


def fake_message(*, content="測試", author_id=100, is_bot=False, guild_id=1, reference=None):
    return SimpleNamespace(
        id=1,
        content=content,
        guild=SimpleNamespace(id=guild_id) if guild_id else None,
        channel=SimpleNamespace(id=10),
        author=SimpleNamespace(id=author_id, bot=is_bot),
        mentions=[],
        reference=reference,
    )


class TestBuildContext:
    @pytest.fixture
    def cog(self, tmp_path):
        bot = MagicMock()
        bot.user = SimpleNamespace(id=999)
        return AutoReply(bot)

    def test_detects_self(self, cog):
        assert cog._build_context(fake_message(author_id=999)).is_self is True

    def test_detects_other_bots(self, cog):
        assert cog._build_context(fake_message(is_bot=True)).author_is_bot is True

    def test_dm_counts_as_a_direct_mention(self, cog):
        ctx = cog._build_context(fake_message(guild_id=None))
        assert ctx.guild_id is None
        assert ctx.mentions_bot is True

    def test_guild_message_without_mention(self, cog):
        assert cog._build_context(fake_message()).mentions_bot is False

    def test_explicit_mention(self, cog):
        message = fake_message()
        message.mentions = [SimpleNamespace(id=999)]
        assert cog._build_context(message).mentions_bot is True

    def test_mention_of_someone_else_does_not_count(self, cog):
        message = fake_message()
        message.mentions = [SimpleNamespace(id=555)]
        assert cog._build_context(message).mentions_bot is False

    def test_reply_to_the_bot_is_detected(self, cog):
        replied_to = MagicMock(spec=discord.Message)
        replied_to.author = SimpleNamespace(id=999)
        reference = SimpleNamespace(resolved=replied_to, cached_message=None)
        assert cog._build_context(fake_message(reference=reference)).is_reply_to_bot is True

    def test_reply_to_a_human_is_not(self, cog):
        replied_to = MagicMock(spec=discord.Message)
        replied_to.author = SimpleNamespace(id=123)
        reference = SimpleNamespace(resolved=replied_to, cached_message=None)
        assert cog._build_context(fake_message(reference=reference)).is_reply_to_bot is False

    def test_deleted_reference_does_not_crash(self, cog):
        reference = SimpleNamespace(resolved=None, cached_message=None)
        assert cog._build_context(fake_message(reference=reference)).is_reply_to_bot is False


CANDIDATE = Candidate(
    label="不可能吧", quote="不可能吧", image_url="https://img/1.JPG", probability=87.51
)


class TestRender:
    def test_image_style_posts_the_bare_url(self):
        assert build_reply(CANDIDATE, "image") == {"content": "https://img/1.JPG"}

    def test_text_style_posts_the_quote(self):
        assert build_reply(CANDIDATE, "text") == {"content": "不可能吧"}

    def test_embed_style_carries_image_and_confidence(self):
        embed = build_reply(CANDIDATE, "embed")["embed"]
        assert embed.description == "不可能吧"
        assert embed.image.url == "https://img/1.JPG"
        assert "87.5%" in embed.footer.text

    def test_image_style_falls_back_to_quote_when_url_missing(self):
        candidate = Candidate(label="x", quote="沒圖", image_url="", probability=1.0)
        assert build_reply(candidate, "image") == {"content": "沒圖"}

    def test_candidates_embed_lists_every_result(self):
        embed = build_candidates_embed([CANDIDATE, CANDIDATE], "好累")
        assert len(embed.fields) == 2
        assert embed.fields[0].name.startswith("1. ")
        assert embed.image.url == "https://img/1.JPG"

    def test_candidates_embed_escapes_user_text(self):
        embed = build_candidates_embed([CANDIDATE], "**不是粗體**")
        assert "**不是粗體**" not in embed.description
