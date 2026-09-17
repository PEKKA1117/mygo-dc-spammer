"""Regression tests for the issues raised in code review on PR #1.

Each test names the defect it pins down so a future refactor cannot quietly
reintroduce it.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.config import Config
from bot.engine import Candidate, RandomEngine
from bot.responder import MessageContext, ReplyPolicy, Trigger
from bot.settings import GuildSettings, SettingsStore
from tests.test_cogs import fake_message, make_config


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, low: float, high: float) -> float:
        return self.value


class TestMalformedSettingsFile:
    """`{"guilds": []}` used to reach .items() and take startup down."""

    def test_guilds_as_a_list_does_not_crash_load(self, tmp_path):
        path = tmp_path / "guilds.json"
        path.write_text('{"version": 1, "guilds": []}', encoding="utf-8")
        store = SettingsStore(path)
        store.load()
        assert store.get(1) == GuildSettings()

    def test_top_level_list_does_not_crash_load(self, tmp_path):
        path = tmp_path / "guilds.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        store = SettingsStore(path)
        store.load()
        assert store.get(1) == GuildSettings()

    def test_guilds_as_a_string_does_not_crash_load(self, tmp_path):
        path = tmp_path / "guilds.json"
        path.write_text('{"guilds": "nope"}', encoding="utf-8")
        store = SettingsStore(path)
        store.load()
        assert store.get(1) == GuildSettings()


class TestBooleanCoercion:
    """A hand-edited "false" is truthy in Python and used to invert the setting."""

    @pytest.mark.parametrize("raw", ["false", "False", "no", "off", "0", ""])
    def test_falsey_strings_disable(self, raw):
        assert GuildSettings.from_dict({"enabled": raw}).enabled is False

    @pytest.mark.parametrize("raw", ["true", "TRUE", "yes", "on", "1"])
    def test_truthy_strings_enable(self, raw):
        assert GuildSettings.from_dict({"enabled": raw}).enabled is True

    def test_integers_coerce(self):
        assert GuildSettings.from_dict({"reply_to_mentions": 0}).reply_to_mentions is False

    def test_unrecognizable_value_falls_back_to_defaults(self):
        settings = GuildSettings.from_dict({"enabled": "maybe?"})
        assert settings.enabled is GuildSettings().enabled

    def test_real_bools_are_untouched(self):
        assert GuildSettings.from_dict({"enabled": False}).enabled is False


class TestWriteSnapshot:
    """Serializing inside the worker thread raced `get()` adding new guilds."""

    def test_snapshot_is_detached_from_live_state(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        store.get(1)
        payload = store._snapshot()
        store.get(2)  # a message from a new guild lands mid-write
        assert set(payload["guilds"]) == {"1"}

    @pytest.mark.asyncio
    async def test_new_guild_during_a_slow_save_does_not_break_it(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        for guild_id in range(50):
            store.get(guild_id)

        real_write = store._write

        def slow_write(payload):
            time.sleep(0.05)  # hold the worker thread open
            real_write(payload)

        store._write = slow_write
        task = asyncio.create_task(store.save())
        await asyncio.sleep(0.01)
        for guild_id in range(50, 100):  # would mutate the dict mid-iteration
            store.get(guild_id)
        await task  # previously: RuntimeError: dictionary changed size

        assert (tmp_path / "guilds.json").exists()


class TestAtomicMutation:
    """Reading a list outside the lock let concurrent edits drop one another."""

    @pytest.mark.asyncio
    async def test_concurrent_keyword_adds_both_survive(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")

        def add(word):
            def mutator(settings):
                return {"keywords": [*settings.keywords, word]}

            return mutator

        await asyncio.gather(
            store.mutate(1, add("春日影")),
            store.mutate(1, add("mygo")),
            store.mutate(1, add("睡不著")),
        )
        assert set(store.get(1).keywords) == {"春日影", "mygo", "睡不著"}

    @pytest.mark.asyncio
    async def test_mutator_sees_changes_from_the_previous_mutation(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        await store.mutate(1, lambda s: {"keywords": ["a"]})
        await store.mutate(1, lambda s: {"keywords": [*s.keywords, "b"]})
        assert store.get(1).keywords == ["a", "b"]

    @pytest.mark.asyncio
    async def test_mutate_still_rejects_unknown_fields(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        with pytest.raises(KeyError):
            await store.mutate(1, lambda s: {"nope": 1})


class TestHundredPercentChance:
    """uniform() can return its upper endpoint, so `< 100` could miss."""

    def test_full_chance_fires_even_on_the_upper_endpoint(self):
        policy = ReplyPolicy(rng=FixedRandom(100.0))
        settings = GuildSettings(reply_chance=100.0)
        ctx = MessageContext(1, 10, 100, "今天好累喔")
        decision = policy.decide(ctx, settings, now=0.0)
        assert decision.should_reply
        assert decision.trigger is Trigger.CHANCE

    def test_zero_chance_still_never_fires(self):
        policy = ReplyPolicy(rng=FixedRandom(0.0))
        settings = GuildSettings(reply_chance=0.0)
        ctx = MessageContext(1, 10, 100, "今天好累喔")
        assert not policy.decide(ctx, settings, now=0.0).should_reply


class TestCooldownRelease:
    """A claimed slot must be handed back when the bot never speaks."""

    def test_release_restores_the_previous_timestamp(self):
        policy = ReplyPolicy()
        policy.record_reply(10, now=5.0)
        previous = policy.record_reply(10, now=9.0)
        assert previous == 5.0
        policy.release_reply(10, previous)
        assert policy.seconds_until_ready(10, cooldown=30.0, now=9.0) == pytest.approx(26.0)

    def test_release_clears_a_first_ever_claim(self):
        policy = ReplyPolicy()
        previous = policy.record_reply(10, now=5.0)
        assert previous is None
        policy.release_reply(10, previous)
        assert policy.seconds_until_ready(10, cooldown=30.0, now=5.0) == 0.0

    def test_record_reply_still_blocks_while_held(self):
        policy = ReplyPolicy()
        policy.record_reply(10, now=0.0)
        assert policy.seconds_until_ready(10, cooldown=30.0, now=1.0) == pytest.approx(29.0)


class TestCandidateCountValidation:
    """CANDIDATE_COUNT=0 made every prediction come back empty, silently."""

    @pytest.mark.parametrize("value", ["0", "-1", "26"])
    def test_out_of_range_is_rejected(self, monkeypatch, value):
        monkeypatch.setenv("DISCORD_TOKEN", "x")
        monkeypatch.setenv("CANDIDATE_COUNT", value)
        with pytest.raises(ValueError, match="CANDIDATE_COUNT"):
            Config.from_env()

    def test_valid_value_is_accepted(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "x")
        monkeypatch.setenv("CANDIDATE_COUNT", "3")
        assert Config.from_env().candidate_count == 3

    def test_zero_max_input_chars_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "x")
        monkeypatch.setenv("MAX_INPUT_CHARS", "0")
        with pytest.raises(ValueError, match="MAX_INPUT_CHARS"):
            Config.from_env()


def fake_interaction(user_id=1, manage_guild=False, in_guild=True):
    interaction = MagicMock()
    interaction.guild = MagicMock() if in_guild else None
    interaction.guild_id = 99 if in_guild else None
    interaction.user.id = user_id
    interaction.user.guild_permissions.manage_guild = manage_guild
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


@pytest.fixture
async def loaded_bot(tmp_path):
    from bot.client import MyGoBot

    bot = MyGoBot(make_config(tmp_path), engine=RandomEngine(seed=1))
    await bot.load_extension("bot.cogs.commands")
    yield bot
    await bot.http.close()


class TestRuntimePermissionCheck:
    """default_permissions is a UI default; admins can reassign the command."""

    @pytest.mark.asyncio
    async def test_member_without_manage_guild_is_rejected(self, loaded_bot):
        cog = loaded_bot.get_cog("mygoconfig")
        with pytest.raises(discord.app_commands.MissingPermissions):
            await cog.interaction_check(fake_interaction(manage_guild=False))

    @pytest.mark.asyncio
    async def test_member_with_manage_guild_passes(self, loaded_bot):
        cog = loaded_bot.get_cog("mygoconfig")
        assert await cog.interaction_check(fake_interaction(manage_guild=True)) is True

    @pytest.mark.asyncio
    async def test_configured_owner_bypasses(self, tmp_path):
        from bot.client import MyGoBot

        config = Config(
            token="x",
            engine="random",
            mygochat_path=tmp_path / "vendor",
            settings_path=tmp_path / "guilds.json",
            max_input_chars=300,
            candidate_count=5,
            owner_ids={4242},
        )
        bot = MyGoBot(config, engine=RandomEngine(seed=1))
        await bot.load_extension("bot.cogs.commands")
        cog = bot.get_cog("mygoconfig")
        assert await cog.interaction_check(fake_interaction(user_id=4242)) is True
        await bot.http.close()

    @pytest.mark.asyncio
    async def test_dm_is_rejected(self, loaded_bot):
        cog = loaded_bot.get_cog("mygoconfig")
        with pytest.raises(discord.app_commands.NoPrivateMessage):
            await cog.interaction_check(fake_interaction(in_guild=False))

    @pytest.mark.asyncio
    async def test_discord_actually_runs_the_check(self, loaded_bot):
        """Covers the wiring, not just the method: discord.py's own dispatch
        path reads `interaction_check` off the cog before invoking a command."""
        group = discord.utils.get(loaded_bot.tree.get_commands(), name="mygoconfig")
        command = discord.utils.get(group.commands, name="enable")
        with pytest.raises(discord.app_commands.MissingPermissions):
            await command._check_can_run(fake_interaction(manage_guild=False))

    @pytest.mark.asyncio
    async def test_public_commands_are_not_gated(self, loaded_bot):
        group = discord.utils.get(loaded_bot.tree.get_commands(), name="mygo")
        command = discord.utils.get(group.commands, name="chat")
        assert await command._check_can_run(fake_interaction(manage_guild=False)) is True


class TestPrivateResultsStayPrivate:
    """defer(ephemeral=True) does not make the follow-up message ephemeral."""

    @pytest.mark.asyncio
    async def test_private_true_sends_an_ephemeral_followup(self, loaded_bot):
        cog = loaded_bot.get_cog("mygo")
        interaction = fake_interaction()
        await cog.chat.callback(cog, interaction, text="今天好累喔", private=True)
        assert interaction.followup.send.call_args.kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_private_false_sends_publicly(self, loaded_bot):
        cog = loaded_bot.get_cog("mygo")
        interaction = fake_interaction()
        await cog.chat.callback(cog, interaction, text="今天好累喔", private=False)
        assert interaction.followup.send.call_args.kwargs["ephemeral"] is False


class SilentEngine:
    """Returns only low-confidence candidates, so nothing clears the bar."""

    name = "silent"

    async def load(self):
        return None

    async def predict(self, text, k=5):
        return [Candidate(label="x", quote="x", image_url="https://img/1.JPG", probability=1.0)]

    async def close(self):
        return None


class TypingChannel:
    def __init__(self, channel_id=10):
        self.id = channel_id

    def typing(self):
        class _CM:
            async def __aenter__(self_inner):
                return None

            async def __aexit__(self_inner, *exc):
                return False

        return _CM()


class TestAutoReplyReleasesCooldown:
    """The slot is claimed before inference; a silent outcome must return it."""

    @staticmethod
    def _bot_with(tmp_path, engine):
        from bot.client import MyGoBot

        bot = MyGoBot(make_config(tmp_path), engine=engine)
        bot._connection.user = SimpleNamespace(id=999)
        # A keyword trigger, so min_confidence applies. A mention would not:
        # direct address deliberately bypasses the confidence bar.
        bot.settings.get(1).keywords = ["累"]
        return bot

    @staticmethod
    def _message():
        message = fake_message(content="今天好累喔")
        message.channel = TypingChannel()
        message.reply = AsyncMock()
        return message

    @pytest.mark.asyncio
    async def test_no_confident_candidate_frees_the_channel(self, tmp_path):
        from bot.cogs.autoreply import AutoReply

        bot = self._bot_with(tmp_path, SilentEngine())
        message = self._message()

        await AutoReply(bot).on_message(message)

        message.reply.assert_not_awaited()
        # Previously the channel stayed muted for the full cooldown.
        assert bot.policy.seconds_until_ready(10, cooldown=30.0, now=time.monotonic()) == 0.0
        await bot.http.close()

    @pytest.mark.asyncio
    async def test_engine_failure_frees_the_channel(self, tmp_path):
        from bot.cogs.autoreply import AutoReply
        from bot.engine import EngineError

        class BrokenEngine(SilentEngine):
            async def predict(self, text, k=5):
                raise EngineError("cuda is on fire")

        bot = self._bot_with(tmp_path, BrokenEngine())
        message = self._message()

        await AutoReply(bot).on_message(message)

        message.reply.assert_not_awaited()
        assert bot.policy.seconds_until_ready(10, cooldown=30.0, now=time.monotonic()) == 0.0
        await bot.http.close()

    @pytest.mark.asyncio
    async def test_an_actual_reply_still_holds_the_channel(self, tmp_path):
        from bot.cogs.autoreply import AutoReply

        class ConfidentEngine(SilentEngine):
            async def predict(self, text, k=5):
                return [
                    Candidate(
                        label="x", quote="x", image_url="https://img/1.JPG", probability=99.0
                    )
                ]

        bot = self._bot_with(tmp_path, ConfidentEngine())
        message = self._message()

        await AutoReply(bot).on_message(message)

        message.reply.assert_awaited_once()
        assert bot.policy.seconds_until_ready(10, cooldown=30.0, now=time.monotonic()) > 0.0
        await bot.http.close()
