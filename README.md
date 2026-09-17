# mygo-dc-spammer

A Discord bot that reads what people are saying and answers with the MyGO!!!!!
screenshot that fits, using [MyGOChat](https://github.com/qaz45647/MyGOChat) —
a RoBERTa classifier that maps a line of Traditional Chinese onto one of 157
captioned images.

```
someone:  期末報告還沒寫完
bot:      [不可能吧.JPG]
```

It replies when @mentioned or replied to, when a message contains a configured
keyword, and — if you turn it on — at random on a percentage of everything else.
Rate limits and a confidence floor are on by default so it stays funny instead of
becoming noise.

---

## Setup

### 1. Create the Discord application

1. Go to https://discord.com/developers/applications → **New Application**.
2. **Bot** → **Reset Token**, copy it.
3. On the same page, enable **Message Content Intent** under *Privileged Gateway
   Intents*. Without it the bot receives empty message bodies and auto-reply
   silently never fires.
4. **OAuth2 → URL Generator**: scopes `bot` + `applications.commands`,
   permissions `Send Messages`, `Read Message History`, `Embed Links`,
   `Use External Emojis`. Open the generated URL to invite it.

### 2. Install

```bash
git clone https://github.com/pekka1117/mygo-dc-spammer.git
cd mygo-dc-spammer
python3 -m venv .venv && source .venv/bin/activate

./scripts/setup.sh          # deps + clones MyGOChat + pulls the 1.3 GB model
```

`scripts/setup.sh` needs **git-lfs** installed (`apt-get install git-lfs` /
`brew install git-lfs`). The checkpoint is an LFS object, and without it you get
a 130-byte pointer file that fails to load later with a confusing error — the
script checks for exactly that.

In a hurry? `SKIP_MODEL=1 ./scripts/setup.sh` skips torch and the download
entirely; set `MYGO_ENGINE=random` and the bot replies with random images. Good
for verifying permissions and commands before committing to the big download.

### 3. Configure and run

```bash
cp .env.example .env
$EDITOR .env                # paste DISCORD_TOKEN
python -m bot
```

Set `DEV_GUILD_ID` to your server's ID while developing: slash commands appear
instantly instead of waiting up to an hour for global propagation.

### Docker

```bash
cp .env.example .env        # fill in DISCORD_TOKEN
docker compose up -d --build
```

The build clones the model in a separate stage and installs the CPU-only torch
wheel, so the image stays far smaller than a default CUDA install. `./data` is
mounted so per-guild settings survive rebuilds.

---

## Commands

| Command | Who | What |
| --- | --- | --- |
| `/mygo chat <text> [private]` | everyone | Best matching image for `text` |
| `/mygo candidates <text> [count]` | everyone | Top-k guesses with confidences |
| `/mygo status` | everyone | Engine, latency, this server's settings |
| `/mygoconfig show` | Manage Server | Full settings dump |
| `/mygoconfig enable <bool>` | Manage Server | Master switch for auto-reply |
| `/mygoconfig mentions <bool>` | Manage Server | Answer @mentions and replies |
| `/mygoconfig chance <percent>` | Manage Server | Odds of butting into other messages |
| `/mygoconfig cooldown <seconds>` | Manage Server | Quiet period per channel |
| `/mygoconfig confidence <percent>` | Manage Server | Confidence floor for ambient replies |
| `/mygoconfig style <image\|embed\|text>` | Manage Server | How replies are rendered |
| `/mygoconfig channel <action> [channel]` | Manage Server | Allowlist / blocklist channels |
| `/mygoconfig ignoreuser <action> <user>` | Manage Server | Mute the bot for one person |
| `/mygoconfig keyword <action> [word]` | Manage Server | Words that always trigger a reply |

`/mygoconfig` is gated on the **Manage Server** permission by Discord itself, so
it never shows up for regular members.

## How it decides to reply

`bot/responder.py` holds the whole policy, with no discord.py imports, so it is
directly testable. In order:

1. Skip the bot's own messages and every other bot.
2. Skip if disabled for the guild, or the author/channel is ignored, or a channel
   allowlist exists and this channel is not in it.
3. Strip mentions, custom emoji and URLs; skip if less than 2 characters remain.
4. Pick a trigger: **mention** (@mentioned, replied to, or any DM) → **keyword**
   → **chance** (a roll against `reply_chance`).
5. Enforce the per-channel cooldown. Mentions bypass `cooldown_seconds` but still
   respect a 3-second floor, so the bot cannot be used to flood a channel.
6. Run the model, then take the highest-confidence candidate that has an image
   and clears the bar — `min_confidence` for ambient replies, **0 for mentions**,
   because someone who directly asks always deserves an answer.

Defaults are deliberately quiet: `reply_chance=0`, so out of the box the bot only
speaks when spoken to. Turn on ambient replies with
`/mygoconfig chance 5` and `/mygoconfig confidence 60`.

## Configuration reference

Environment (see `.env.example`): `DISCORD_TOKEN`, `MYGO_ENGINE`,
`MYGOCHAT_PATH`, `SETTINGS_PATH`, `MAX_INPUT_CHARS`, `CANDIDATE_COUNT`,
`OWNER_IDS`, `DEV_GUILD_ID`, `LOG_LEVEL`.

Per-guild settings live in `data/guilds.json`, written atomically
(temp file + rename) so a crash mid-write cannot corrupt them.

## Layout

```
bot/
  __main__.py     entry point, logging, env loading
  config.py       environment -> Config
  client.py       MyGoBot: intents, cog loading, command sync, error handling
  engine.py       MyGOChatEngine (threaded) and RandomEngine
  responder.py    when-to-reply policy (pure, no discord imports)
  settings.py     per-guild settings + atomic JSON persistence
  render.py       Candidate -> Discord message/embed
  cogs/
    autoreply.py  the on_message listener
    commands.py   /mygo and /mygoconfig
docs/MYGOCHAT.md  upstream model notes and the traps it sets
scripts/setup.sh  dependency install + LFS-aware model fetch
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite runs without torch and without the checkpoint — it covers the reply
policy, settings persistence, the engine wrappers and the discord.py-facing
layer. CI runs it on Python 3.10, 3.11 and 3.12.

## Notes and credits

- The model is Traditional Chinese only and reports ~86% accuracy upstream.
  English input mostly produces nonsense, which is arguably the point.
- Images are hotlinked from bahamut.com.tw, exactly as upstream ships them.
- Model and images: [qaz45647/MyGOChat](https://github.com/qaz45647/MyGOChat).
  See `docs/MYGOCHAT.md` for integration details.
- Despite the repo name, please do not actually spam. `cooldown_seconds`,
  `min_confidence` and a default `reply_chance` of 0 exist for a reason; a bot
  that posts on every message will get muted by your members and rate-limited by
  Discord.
