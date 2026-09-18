# Working with MyGOChat

Notes on the upstream model this bot wraps:
[qaz45647/MyGOChat](https://github.com/qaz45647/MyGOChat).

## What it actually is

A RoBERTa (`hfl/chinese-roberta-wwm-ext-large`) sequence classifier fine-tuned
over **157 labels**, where each label is a MyGO!!!!! screenshot with its caption.
It is *not* a generative model and *not* a web service — there is no hosted API
endpoint. You feed it a line of text, it returns a probability distribution over
those 157 images.

The public interface is two methods:

```python
chat.chat(text)                      # -> {"quote": ..., "image_url": ...}
chat.chat_with_candidates(text, k=5) # -> {"top_prediction": ..., "candidates": [...]}
```

Each candidate carries `label`, `probability` (already a 0–100 percentage),
`quote` and `image_url`. `bot/engine.py` only calls `chat_with_candidates`,
since `chat()` is just `k=1` with the confidence discarded — and this bot needs
the confidence to decide whether butting in is worth it.

## Three things that will bite you

**1. The checkpoint is a git-lfs object.**
`mygochat/models/model.safetensors` is 1.3 GB and stored in LFS. A plain
`git clone` on a machine without git-lfs produces a ~130-byte pointer file that
starts with `version https://git-lfs.github.com/spec/v1`. `from_pretrained()`
then fails with an opaque deserialization error. `scripts/setup.sh` checks the
file size after cloning specifically to catch this.

**2. There is no `setup.py`.**
`pip install git+https://github.com/qaz45647/MyGOChat.git` does not work. The
package has to be imported off the filesystem, which is why `MyGOChatEngine`
inserts `MYGOCHAT_PATH` into `sys.path` before `import mygochat`.

**3. The upstream pins are old.**
`requirements.txt` upstream says `torch==2.1.0` / `transformers==4.30.0`, and
the README suggests `conda create -n MyGO_env python=3.9`. Those versions have
no wheels for Python 3.11+. This repo loosens them to `torch>=2.1` /
`transformers>=4.30`, which loads the same checkpoint without complaint. If you
hit a genuine incompatibility, reproduce the original environment:

```bash
conda create -n MyGO_env python=3.9 && conda activate MyGO_env
pip install torch==2.1.0 transformers==4.30.0
```

Note also that `chatbot.py` calls `torch.cuda.amp.autocast()` unconditionally.
On a CPU-only host this is a no-op that emits a `UserWarning` on every
prediction — harmless, but it explains the log noise.

## Threading

`MyGOChat` is synchronous and holds torch modules that are not safe to use from
several threads at once. Blocking the event loop on a multi-second inference
would stall the gateway heartbeat and get the bot disconnected, so
`MyGOChatEngine` dispatches every call to a `ThreadPoolExecutor(max_workers=1)`:
the single worker keeps inference serialized while the loop stays responsive.
First call is slow (model load); later calls are fast.

## Accuracy

Upstream reports ~86% accuracy and documents the limits: Traditional Chinese
only, no image input, weak multi-turn handling. This is why the bot has a
`min_confidence` setting — ambient replies are gated on the model being fairly
sure, while a direct @mention always gets an answer (see
`required_confidence()` in `bot/responder.py`).
