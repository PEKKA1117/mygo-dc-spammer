import json

import pytest

from bot.engine import Candidate, EngineError, MyGOChatEngine, RandomEngine


class TestCandidate:
    def test_from_payload_maps_mygochat_fields(self):
        candidate = Candidate.from_payload(
            {"label": "不可能吧", "probability": 87.5, "quote": "不可能吧", "image_url": "u"}
        )
        assert candidate.quote == "不可能吧"
        assert candidate.probability == 87.5

    def test_from_payload_tolerates_missing_fields(self):
        candidate = Candidate.from_payload({"label": "x"})
        assert candidate.quote == "x"
        assert candidate.image_url == ""
        assert candidate.probability == 0.0


def write_labels(root, rows):
    data_dir = root / "mygochat" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "Label_Path.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


class TestRandomEngine:
    @pytest.mark.asyncio
    async def test_reads_quotes_from_a_checkout(self, tmp_path):
        write_labels(
            tmp_path,
            [
                {"title": "一丘之貂", "Image_Path": "https://img/1.JPG"},
                {"title": "不可能吧", "Image_Path": "https://img/2.JPG"},
            ],
        )
        engine = RandomEngine(mygochat_path=tmp_path, seed=1)
        results = await engine.predict("測試", k=2)
        assert {c.quote for c in results} == {"一丘之貂", "不可能吧"}

    @pytest.mark.asyncio
    async def test_falls_back_when_checkout_is_missing(self, tmp_path):
        engine = RandomEngine(mygochat_path=tmp_path / "absent", seed=1)
        results = await engine.predict("測試", k=5)
        assert results and all(c.image_url for c in results)

    @pytest.mark.asyncio
    async def test_falls_back_on_corrupt_label_file(self, tmp_path):
        data_dir = tmp_path / "mygochat" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "Label_Path.json").write_text("{broken", encoding="utf-8")
        assert await RandomEngine(mygochat_path=tmp_path, seed=1).predict("測試")

    @pytest.mark.asyncio
    async def test_probabilities_are_descending(self, tmp_path):
        engine = RandomEngine(pairs=[(f"q{i}", f"u{i}") for i in range(5)], seed=7)
        results = await engine.predict("測試", k=5)
        probabilities = [c.probability for c in results]
        assert probabilities == sorted(probabilities, reverse=True)

    @pytest.mark.asyncio
    async def test_k_larger_than_corpus_is_clamped(self):
        engine = RandomEngine(pairs=[("a", "u")], seed=1)
        assert len(await engine.predict("測試", k=10)) == 1

    @pytest.mark.asyncio
    async def test_empty_input_returns_nothing(self):
        engine = RandomEngine(pairs=[("a", "u")], seed=1)
        assert await engine.predict("   ") == []

    @pytest.mark.asyncio
    async def test_seeded_runs_are_reproducible(self):
        pairs = [(f"q{i}", f"u{i}") for i in range(20)]
        first = await RandomEngine(pairs=pairs, seed=3).predict("x", k=3)
        second = await RandomEngine(pairs=pairs, seed=3).predict("x", k=3)
        assert first == second


class TestMyGOChatEngine:
    @pytest.mark.asyncio
    async def test_missing_checkout_raises_actionable_error(self, tmp_path):
        engine = MyGOChatEngine(mygochat_path=tmp_path / "absent")
        with pytest.raises(EngineError, match="scripts/setup.sh"):
            await engine.load()
        await engine.close()

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits_before_loading(self, tmp_path):
        engine = MyGOChatEngine(mygochat_path=tmp_path / "absent")
        assert await engine.predict("") == []
        await engine.close()

    @pytest.mark.asyncio
    async def test_input_is_truncated_to_the_configured_limit(self, tmp_path):
        engine = MyGOChatEngine(mygochat_path=tmp_path, max_input_chars=5)
        seen = {}

        def fake_predict(text, k):
            seen["text"] = text
            return [{"label": "x", "quote": "x", "image_url": "u", "probability": 1.0}]

        engine._chat = object()
        engine._predict_blocking = fake_predict  # type: ignore[method-assign]
        await engine.predict("一二三四五六七八九十", k=1)
        assert seen["text"] == "一二三四五"
        await engine.close()

    @pytest.mark.asyncio
    async def test_wraps_inference_failures(self, tmp_path):
        engine = MyGOChatEngine(mygochat_path=tmp_path)
        engine._chat = object()

        def boom(text, k):
            raise ValueError("cuda is on fire")

        engine._predict_blocking = boom  # type: ignore[method-assign]
        with pytest.raises(EngineError, match="cuda is on fire"):
            await engine.predict("測試")
        await engine.close()
