import pytest

from bot.responder import (
    MENTION_COOLDOWN_FLOOR,
    Decision,
    MessageContext,
    ReplyPolicy,
    Trigger,
    clean_content,
    pick_candidate,
    required_confidence,
)
from bot.engine import Candidate
from bot.settings import GuildSettings


class FixedRandom:
    """random.Random stand-in that always returns the value it was given."""

    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, low: float, high: float) -> float:
        return self.value


def ctx(**overrides) -> MessageContext:
    base = dict(guild_id=1, channel_id=10, author_id=100, content="今天好累喔")
    base.update(overrides)
    return MessageContext(**base)


def always_replies() -> ReplyPolicy:
    return ReplyPolicy(rng=FixedRandom(0.0))


def never_by_chance() -> ReplyPolicy:
    return ReplyPolicy(rng=FixedRandom(100.0))


class TestCleanContent:
    def test_strips_mentions_emoji_and_urls(self):
        raw = "<@123> 看看這個 https://example.com/x <a:pien:456> 很扯"
        assert clean_content(raw) == "看看這個 很扯"

    def test_collapses_whitespace(self):
        assert clean_content("  好   累\n\n喔 ") == "好 累 喔"

    def test_mention_only_message_becomes_empty(self):
        assert clean_content("<@123>") == ""


class TestSkipRules:
    def test_ignores_own_messages(self):
        decision = never_by_chance().decide(ctx(is_self=True), GuildSettings(), now=0.0)
        assert decision == Decision.skip("own message")

    def test_ignores_other_bots(self):
        decision = always_replies().decide(
            ctx(author_is_bot=True, mentions_bot=True), GuildSettings(), now=0.0
        )
        assert not decision.should_reply

    def test_disabled_guild_ignores_even_mentions(self):
        settings = GuildSettings(enabled=False)
        decision = always_replies().decide(ctx(mentions_bot=True), settings, now=0.0)
        assert not decision.should_reply
        assert decision.reason == "disabled in this guild"

    def test_ignored_user(self):
        settings = GuildSettings(ignored_users=[100])
        assert not always_replies().decide(ctx(mentions_bot=True), settings, 0.0).should_reply

    def test_ignored_channel(self):
        settings = GuildSettings(ignored_channels=[10])
        assert not always_replies().decide(ctx(mentions_bot=True), settings, 0.0).should_reply

    def test_allowlist_excludes_other_channels(self):
        settings = GuildSettings(channels=[999])
        assert not always_replies().decide(ctx(mentions_bot=True), settings, 0.0).should_reply

    def test_allowlist_includes_listed_channel(self):
        settings = GuildSettings(channels=[10])
        assert always_replies().decide(ctx(mentions_bot=True), settings, 0.0).should_reply

    def test_message_with_no_usable_text(self):
        decision = always_replies().decide(ctx(content="<@1>", mentions_bot=True), GuildSettings(), 0.0)
        assert decision.reason == "not enough text"


class TestTriggers:
    def test_mention_replies_even_at_zero_chance(self):
        settings = GuildSettings(reply_chance=0.0)
        decision = never_by_chance().decide(ctx(mentions_bot=True), settings, 0.0)
        assert decision.should_reply
        assert decision.trigger is Trigger.MENTION

    def test_reply_to_bot_counts_as_mention(self):
        decision = never_by_chance().decide(ctx(is_reply_to_bot=True), GuildSettings(), 0.0)
        assert decision.trigger is Trigger.MENTION

    def test_mentions_can_be_switched_off(self):
        settings = GuildSettings(reply_to_mentions=False, reply_chance=0.0)
        assert not never_by_chance().decide(ctx(mentions_bot=True), settings, 0.0).should_reply

    def test_keyword_forces_a_reply(self):
        settings = GuildSettings(keywords=["春日影"], reply_chance=0.0)
        decision = never_by_chance().decide(ctx(content="為什麼要演奏春日影！"), settings, 0.0)
        assert decision.trigger is Trigger.KEYWORD

    def test_keyword_matching_is_case_insensitive(self):
        settings = GuildSettings(keywords=["mygo"], reply_chance=0.0)
        decision = never_by_chance().decide(ctx(content="I love MyGO so much"), settings, 0.0)
        assert decision.trigger is Trigger.KEYWORD

    def test_chance_roll_under_threshold_replies(self):
        settings = GuildSettings(reply_chance=10.0)
        decision = ReplyPolicy(rng=FixedRandom(9.99)).decide(ctx(), settings, 0.0)
        assert decision.trigger is Trigger.CHANCE

    def test_chance_roll_over_threshold_stays_quiet(self):
        settings = GuildSettings(reply_chance=10.0)
        assert not ReplyPolicy(rng=FixedRandom(10.01)).decide(ctx(), settings, 0.0).should_reply

    def test_zero_chance_never_rolls(self):
        settings = GuildSettings(reply_chance=0.0)
        decision = ReplyPolicy(rng=FixedRandom(-1.0)).decide(ctx(), settings, 0.0)
        assert not decision.should_reply

    def test_decision_carries_cleaned_text(self):
        decision = always_replies().decide(
            ctx(content="<@9> 好想睡", mentions_bot=True), GuildSettings(), 0.0
        )
        assert decision.text == "好想睡"


class TestCooldown:
    def test_blocks_second_chance_reply_within_window(self):
        policy = always_replies()
        settings = GuildSettings(reply_chance=100.0, cooldown_seconds=30)

        assert policy.decide(ctx(), settings, now=0.0).should_reply
        policy.record_reply(10, now=0.0)

        blocked = policy.decide(ctx(), settings, now=5.0)
        assert not blocked.should_reply
        assert "cooling down" in blocked.reason

    def test_allows_reply_once_window_elapsed(self):
        policy = always_replies()
        settings = GuildSettings(reply_chance=100.0, cooldown_seconds=30)
        policy.record_reply(10, now=0.0)
        assert policy.decide(ctx(), settings, now=30.0).should_reply

    def test_cooldown_is_per_channel(self):
        policy = always_replies()
        settings = GuildSettings(reply_chance=100.0, cooldown_seconds=30)
        policy.record_reply(10, now=0.0)
        assert policy.decide(ctx(channel_id=11), settings, now=1.0).should_reply

    def test_mentions_bypass_the_configured_cooldown(self):
        policy = always_replies()
        settings = GuildSettings(cooldown_seconds=3600)
        policy.record_reply(10, now=0.0)
        assert policy.decide(ctx(mentions_bot=True), settings, now=10.0).should_reply

    def test_mentions_still_respect_the_anti_spam_floor(self):
        policy = always_replies()
        settings = GuildSettings(cooldown_seconds=0)
        policy.record_reply(10, now=0.0)
        blocked = policy.decide(ctx(mentions_bot=True), settings, now=MENTION_COOLDOWN_FLOOR / 2)
        assert not blocked.should_reply

    def test_zero_cooldown_allows_back_to_back_chance_replies(self):
        policy = always_replies()
        settings = GuildSettings(reply_chance=100.0, cooldown_seconds=0)
        policy.record_reply(10, now=0.0)
        assert policy.decide(ctx(), settings, now=0.0).should_reply


class TestCandidateSelection:
    @staticmethod
    def candidate(prob: float, url: str = "https://img/1.jpg") -> Candidate:
        return Candidate(label="x", quote="x", image_url=url, probability=prob)

    def test_mentions_ignore_the_confidence_floor(self):
        settings = GuildSettings(min_confidence=90.0)
        assert required_confidence(Trigger.MENTION, settings) == 0.0

    def test_ambient_triggers_use_the_configured_floor(self):
        settings = GuildSettings(min_confidence=42.0)
        assert required_confidence(Trigger.CHANCE, settings) == 42.0
        assert required_confidence(Trigger.KEYWORD, settings) == 42.0

    def test_picks_first_candidate_above_threshold(self):
        settings = GuildSettings(min_confidence=50.0)
        picked = pick_candidate(
            [self.candidate(20.0), self.candidate(80.0)], Trigger.CHANCE, settings
        )
        assert picked is not None and picked.probability == 80.0

    def test_returns_none_when_nothing_clears_threshold(self):
        settings = GuildSettings(min_confidence=95.0)
        assert pick_candidate([self.candidate(90.0)], Trigger.CHANCE, settings) is None

    def test_skips_candidates_without_an_image(self):
        settings = GuildSettings(min_confidence=0.0)
        picked = pick_candidate(
            [self.candidate(99.0, url=""), self.candidate(10.0)], Trigger.CHANCE, settings
        )
        assert picked is not None and picked.probability == 10.0

    def test_empty_candidate_list(self):
        assert pick_candidate([], Trigger.MENTION, GuildSettings()) is None
