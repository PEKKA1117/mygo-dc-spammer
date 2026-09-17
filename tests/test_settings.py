import json

import pytest

from bot.settings import GuildSettings, SettingsStore


class TestNormalization:
    def test_clamps_percentages(self):
        settings = GuildSettings(reply_chance=150.0, min_confidence=-5.0).normalized()
        assert settings.reply_chance == 100.0
        assert settings.min_confidence == 0.0

    def test_clamps_negative_cooldown(self):
        assert GuildSettings(cooldown_seconds=-10).normalized().cooldown_seconds == 0

    def test_rejects_unknown_style(self):
        assert GuildSettings(style="interpretive-dance").normalized().style == "image"

    def test_lowercases_and_drops_blank_keywords(self):
        settings = GuildSettings(keywords=["  MyGO ", "", "   "]).normalized()
        assert settings.keywords == ["mygo"]

    def test_deduplicates_and_coerces_ids(self):
        settings = GuildSettings(channels=["5", 5, 7, "nope"]).normalized()
        assert settings.channels == [5, 7]


class TestFromDict:
    def test_ignores_unknown_keys(self):
        settings = GuildSettings.from_dict({"enabled": False, "from_the_future": 1})
        assert settings.enabled is False

    def test_falls_back_to_defaults_on_bad_types(self):
        settings = GuildSettings.from_dict({"reply_chance": "a lot"})
        assert settings.reply_chance == GuildSettings().reply_chance

    def test_roundtrips_through_to_dict(self):
        original = GuildSettings(reply_chance=12.5, keywords=["mygo"], channels=[1, 2])
        assert GuildSettings.from_dict(original.to_dict()) == original


class TestSettingsStore:
    def test_missing_file_is_not_an_error(self, tmp_path):
        store = SettingsStore(tmp_path / "nope.json")
        store.load()
        assert store.get(1) == GuildSettings()

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "guilds.json"
        path.write_text("{not json", encoding="utf-8")
        store = SettingsStore(path)
        store.load()
        assert store.get(1).enabled is True

    def test_guilds_are_isolated_from_each_other(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        store.get(1).reply_chance = 50.0
        assert store.get(2).reply_chance == 0.0

    def test_dm_settings_are_a_detached_copy(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        first = store.get(None)
        first.reply_chance = 99.0
        assert store.get(None).reply_chance == 0.0

    @pytest.mark.asyncio
    async def test_update_persists_and_survives_reload(self, tmp_path):
        path = tmp_path / "guilds.json"
        store = SettingsStore(path)
        await store.update(42, reply_chance=25.0, keywords=["春日影"])

        reloaded = SettingsStore(path)
        reloaded.load()
        settings = reloaded.get(42)
        assert settings.reply_chance == 25.0
        assert settings.keywords == ["春日影"]

    @pytest.mark.asyncio
    async def test_update_normalizes_before_writing(self, tmp_path):
        path = tmp_path / "guilds.json"
        store = SettingsStore(path)
        await store.update(42, reply_chance=1000.0)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["guilds"]["42"]["reply_chance"] == 100.0

    @pytest.mark.asyncio
    async def test_update_rejects_unknown_field(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        with pytest.raises(KeyError):
            await store.update(1, definitely_not_a_field=True)

    @pytest.mark.asyncio
    async def test_write_leaves_no_temp_files_behind(self, tmp_path):
        store = SettingsStore(tmp_path / "guilds.json")
        await store.update(1, enabled=False)
        assert [p.name for p in tmp_path.iterdir()] == ["guilds.json"]

    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, tmp_path):
        store = SettingsStore(tmp_path / "nested" / "deep" / "guilds.json")
        await store.update(1, enabled=False)
        assert (tmp_path / "nested" / "deep" / "guilds.json").exists()

    def test_reads_legacy_flat_mapping(self, tmp_path):
        path = tmp_path / "guilds.json"
        path.write_text(json.dumps({"7": {"reply_chance": 5.0}}), encoding="utf-8")
        store = SettingsStore(path)
        store.load()
        assert store.get(7).reply_chance == 5.0
