# mygo-dc-spammer

**繁體中文** · [English](README.md)

一個會讀群裡在聊什麼、然後丟出最貼切那張 MyGO!!!!! 圖的 Discord 機器人。背後用的是
[MyGOChat](https://github.com/qaz45647/MyGOChat) —— 一個把繁體中文句子對應到 157 張
有台詞截圖的 RoBERTa 分類器。

```
某人：期末報告還沒寫完
bot： [不可能吧.JPG]
```

被 @ 或被回覆時會回、訊息裡出現設定好的關鍵字時會回，如果你自己打開的話，也會用一個機率
對其他所有訊息插嘴。冷卻時間和信心度門檻預設就是開著的，讓它維持在好笑的程度而不是變成噪音。

---

## 目錄

- [運作原理](#運作原理)
- [安裝](#安裝)
- [使用方式](#使用方式)
- [指令一覽](#指令一覽)
- [設定參考](#設定參考)
- [開發](#開發)
- [疑難排解](#疑難排解)

---

## 運作原理

### MyGOChat 是什麼（以及不是什麼）

先講清楚，因為這決定了整個架構：MyGOChat **不是**生成式模型，也**不是** web API，
**沒有**任何可以打的 endpoint。

它是一個用 `hfl/chinese-roberta-wwm-ext-large` 微調出來的**文字分類器**。你餵它一句話，
它回傳一個橫跨 **157 個標籤**的機率分佈，每個標籤就是一張 MyGO 截圖配上它的台詞。

所以模型必須跑在你自己的機器上，而不是發 HTTP request 出去。這也是為什麼安裝步驟要下載
1.3 GB 的權重檔。

對外的介面只有兩個方法：

```python
chat.chat(text)                      # -> {"quote": ..., "image_url": ...}
chat.chat_with_candidates(text, k=5) # -> {"top_prediction": ..., "candidates": [...]}
```

`bot/engine.py` 只呼叫 `chat_with_candidates`，因為 `chat()` 本質上就是 `k=1` 然後把
信心度丟掉 —— 而這個 bot 需要信心度來判斷「現在插嘴值不值得」。

### 判斷要不要回話

整套規則都在 `bot/responder.py`，而且**完全不 import discord.py**，所以可以直接對它寫
單元測試，不用去 mock 整個 gateway。順序如下：

1. 跳過自己發的訊息，以及所有其他 bot（不然兩隻 bot 會互相回到天荒地老）。
2. 跳過：該伺服器已關閉、作者或頻道在忽略清單裡、或是有設頻道白名單但這個頻道不在裡面。
3. 去掉 mention、自訂表情、網址之後，剩不到 2 個字就跳過 —— 分類器拿到這種輸入也只是亂猜。
4. 決定觸發來源，依序是：**mention**（被 @、被回覆，或任何私訊）→ **keyword**
   → **chance**（擲骰對上 `reply_chance`）。
5. 檢查該頻道的冷卻時間。
6. 跑模型，從候選裡挑出「有圖 **而且** 信心度過門檻」中分數最高的那個。

其中兩個例外是刻意設計的：

> **被 @ 的時候，`cooldown_seconds` 和 `min_confidence` 都不算數。**
> 有人直接問你，就該給答案，不該因為模型沒把握就裝死。但仍然保留 **3 秒**的每頻道下限，
> 所以沒有人能拿這隻 bot 去洗頻。

> **預設 `reply_chance` 是 0。**
> 雖然 repo 叫 spammer，開箱之後它只在被叫到的時候說話。要它主動插嘴請自己開。

信心度門檻之所以存在，是因為上游自己回報準確率大約 86%，而且只吃繁體中文。沒把握的時候
安靜比亂接話好。

### 為什麼推論要丟到執行緒

MyGOChat 是同步的，而且裡面的 torch module 不能給多個執行緒同時使用。如果直接在 asyncio
的 event loop 上跑推論，那幾秒鐘會卡住 gateway 的 heartbeat，然後機器人就被 Discord 斷線。

`MyGOChatEngine` 因此把每一次呼叫都丟進 `ThreadPoolExecutor(max_workers=1)`。單一 worker
同時解決兩件事：推論被序列化（torch 安全），event loop 保持有回應（連線不斷）。

第一次呼叫會比較慢（要載入模型），之後就快了。

---

## 安裝

### 1. 建立 Discord 應用程式

1. 到 https://discord.com/developers/applications → **New Application**。
2. **Bot** → **Reset Token**，複製下來。
3. 在同一頁，把 *Privileged Gateway Intents* 底下的 **Message Content Intent** 打開。

   > ⚠️ **這步不能跳過。** 沒開的話機器人收到的訊息內容全都是空字串，自動回覆會**安靜地**
   > 完全不作用 —— 不會報錯，就只是永遠不回話。

4. **OAuth2 → URL Generator**：scope 勾 `bot` + `applications.commands`，
   權限勾 `Send Messages`、`Read Message History`、`Embed Links`、`Use External Emojis`。
   打開產生出來的網址把它邀請進伺服器。

### 2. 安裝程式

```bash
git clone https://github.com/PEKKA1117/mygo-dc-spammer.git
cd mygo-dc-spammer
python3 -m venv .venv && source .venv/bin/activate

./scripts/setup.sh          # 裝套件 + clone MyGOChat + 下載 1.3 GB 模型
```

`scripts/setup.sh` 需要系統裝好 **git-lfs**：

```bash
sudo apt-get install git-lfs    # Debian / Ubuntu
brew install git-lfs            # macOS
```

權重檔是 git-lfs 物件。沒裝 git-lfs 的話 clone 會「成功」，但留下來的是一個 130 bytes 的
pointer 檔，等到載入模型時才會爆出看不懂的錯誤。這個腳本 clone 完會檢查檔案大小，就是為了
提早抓到這件事。

**趕時間？**

```bash
SKIP_MODEL=1 ./scripts/setup.sh    # 跳過 torch 和那 1.3 GB
```

然後在 `.env` 設 `MYGO_ENGINE=random`，機器人會隨機挑圖回。適合先把權限、指令、頻道設定
都調通，之後再補模型。

### 3. 設定並啟動

```bash
cp .env.example .env
$EDITOR .env                # 貼上 DISCORD_TOKEN
python -m bot
```

開發的時候把 `DEV_GUILD_ID` 設成你伺服器的 ID：斜線指令會**立刻**出現，不用等全域同步
（最久可能要一小時）。

### Docker

```bash
cp .env.example .env        # 填好 DISCORD_TOKEN
docker compose up -d --build
```

build 分成兩個階段，模型在獨立的 stage 下載，而且裝的是 CPU-only 的 torch wheel，
所以 image 比預設的 CUDA 版小非常多。各伺服器的設定存在名為 `mygo-data` 的 named volume，
重 build 不會消失。

這裡刻意用 named volume 而不是 `./data` bind mount：容器是以非特權的 uid 10001 執行，
而 bind mount 會用宿主機的目錄（通常屬於你自己的帳號）遮蔽掉 image 裡已經設好擁有者的
`/app/data`，結果就是 bot 根本寫不出 `guilds.json`，所有設定變更都會遺失。如果你想把檔案
留在宿主機上方便查看，`docker-compose.yml` 裡的註解寫了要怎麼做。

---

## 使用方式

### 最基本的用法

直接 @ 它，或回覆它的訊息：

```
你：@MyGO 今天好累喔
bot：[你知道這對她造成多大的傷害嗎.PNG]
```

私訊也可以，私訊裡的每一句話都算直接對它說話，不用 @。

### 手動查詢

```
/mygo chat text:期末報告還沒寫完
/mygo chat text:期末報告還沒寫完 private:True     ← 只有你看得到
/mygo candidates text:期末報告還沒寫完 count:5    ← 看前 5 名和信心度
```

`/mygo candidates` 在調 `confidence` 門檻的時候特別有用 —— 可以先看看模型對各種句子
到底有多少把握，再決定門檻要設多少。

### 讓它主動插嘴

預設是關的。要打開：

```
/mygoconfig chance 5          ← 5% 的訊息會被插嘴
/mygoconfig confidence 60     ← 但只在模型有 60% 以上把握時
/mygoconfig cooldown 60       ← 而且同一個頻道 60 秒內最多一次
```

建議從小的數字開始往上加。`chance 5` + `confidence 60` 大概是「偶爾出現、剛好好笑」的
程度；`chance 50` 會讓你的成員把它靜音。

### 限制它只在特定頻道說話

```
/mygoconfig channel action:allow channel:#梗圖
/mygoconfig channel action:allow channel:#閒聊
```

只要白名單非空，其他頻道就完全不會被插嘴。反過來，如果只想擋某幾個頻道：

```
/mygoconfig channel action:ignore channel:#公告
```

### 關鍵字

不管機率多少，只要訊息裡出現這些字就一定回：

```
/mygoconfig keyword action:add word:春日影
```

---

## 指令一覽

### 所有人

| 指令 | 功能 |
| --- | --- |
| `/mygo chat <text> [private]` | 給出最符合這句話的圖 |
| `/mygo candidates <text> [count]` | 列出前 k 名候選與信心度 |
| `/mygo status` | 引擎、延遲、本伺服器目前設定 |

### 需要「管理伺服器」權限

| 指令 | 功能 |
| --- | --- |
| `/mygoconfig show` | 顯示完整設定 |
| `/mygoconfig enable <bool>` | 自動回覆總開關 |
| `/mygoconfig mentions <bool>` | 被 @ 或被回覆時是否一定要回 |
| `/mygoconfig chance <percent>` | 對其他訊息插嘴的機率（0-100） |
| `/mygoconfig cooldown <seconds>` | 同一頻道兩次回覆之間的秒數 |
| `/mygoconfig confidence <percent>` | 低於這個信心度就不插嘴（0-100） |
| `/mygoconfig style <image\|embed\|text>` | 回覆的呈現方式 |
| `/mygoconfig channel <action> [channel]` | 頻道白名單／黑名單 |
| `/mygoconfig ignoreuser <action> <user>` | 對特定使用者靜音 |
| `/mygoconfig keyword <action> [word]` | 管理必定觸發的關鍵字 |

`/mygoconfig` 一般成員在指令選單裡看不到（靠 `default_permissions`），而且執行時還會
再檢查一次權限 —— 因為 `default_permissions` 只是「預設值」，伺服器管理員可以在
**伺服器設定 → 整合** 把這組指令改分配給任何身分組。列在 `OWNER_IDS` 裡的使用者會跳過
這個檢查，用來救那種「管理員把自己權限拿掉」的伺服器。

`channel` 的 action 有：`allow`、`unallow`、`ignore`、`unignore`、`clear`。
`keyword` 的 action 有：`add`、`remove`、`clear`。

---

## 設定參考

### 環境變數（`.env`）

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `DISCORD_TOKEN` | *（必填）* | Developer Portal 拿到的 bot token |
| `MYGO_ENGINE` | `mygochat` | `mygochat` = 真的模型；`random` = 隨機挑圖，不需要 torch |
| `MYGOCHAT_PATH` | `./vendor/MyGOChat` | MyGOChat clone 的位置 |
| `SETTINGS_PATH` | `./data/guilds.json` | 各伺服器設定存放位置 |
| `MAX_INPUT_CHARS` | `300` | 餵給分類器的最長字數 |
| `CANDIDATE_COUNT` | `5` | 每次推論要幾個候選 |
| `OWNER_IDS` | *(空)* | 逗號分隔的使用者 ID；這些人在任何伺服器都能用 `/mygoconfig`，不需要管理伺服器權限 |
| `DEV_GUILD_ID` | *(空)* | 設了就只同步斜線指令到這個伺服器（立即生效） |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 各伺服器設定（存在 `data/guilds.json`）

| 欄位 | 預設 | 說明 |
| --- | --- | --- |
| `enabled` | `true` | 自動回覆總開關 |
| `reply_to_mentions` | `true` | 被 @ 或被回覆時是否一定要回 |
| `reply_chance` | `0.0` | 對其他訊息插嘴的機率（%） |
| `channels` | `[]` | 頻道白名單，空的代表所有頻道 |
| `ignored_channels` | `[]` | 頻道黑名單 |
| `ignored_users` | `[]` | 被忽略的使用者 |
| `keywords` | `[]` | 必定觸發回覆的字串 |
| `cooldown_seconds` | `30` | 每頻道冷卻秒數 |
| `min_confidence` | `35.0` | 插嘴所需的最低信心度（%） |
| `style` | `image` | `image` / `embed` / `text` |

`style` 三種的差別：

- `image` —— 只貼圖片網址，Discord 會直接展開成一張圖，看起來就像有人貼梗圖
- `embed` —— 圖片加上台詞當說明，右下角附信心度
- `text` —— 只有台詞，不貼圖

這個檔案是用「先寫暫存檔再 rename」的方式寫入的，所以寫到一半當掉也不會把設定弄壞。
從磁碟讀回來的值都會經過正規化，手動改壞了也不會讓 bot 啟動失敗。

---

## 開發

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

125 個測試，涵蓋回話判斷邏輯、設定持久化、兩個 engine，以及 discord.py 那一層
（cog 載入、權限閘門、embed 產生）。

**測試不需要 torch，也不需要那份 checkpoint 就能跑** —— CI 也一樣，所以 CI 不用去拉
2 GB 的 wheel。CI 在 Python 3.10、3.11、3.12 上執行。

### 專案結構

```
bot/
  __main__.py     進入點、logging、載入 .env
  config.py       環境變數 -> Config
  client.py       MyGoBot：intents、載入 cog、指令同步、錯誤處理
  engine.py       MyGOChatEngine（丟到執行緒）與 RandomEngine
  responder.py    要不要回話的判斷邏輯（純邏輯，不 import discord）
  settings.py     各伺服器設定 + 原子性 JSON 寫入
  render.py       Candidate -> Discord 訊息／embed
  cogs/
    autoreply.py  on_message 監聽器
    commands.py   /mygo 與 /mygoconfig
docs/MYGOCHAT.md  上游模型的細節，以及它埋的那些坑
scripts/setup.sh  安裝相依套件 + 處理 git-lfs 的模型下載
```

---

## 疑難排解

**機器人上線了，但完全不回話**
八成是 **Message Content Intent** 沒開。去 Developer Portal → 你的應用程式 → Bot →
Privileged Gateway Intents 打開它，然後重啟 bot。沒開的話它收到的訊息內容是空字串，
不會報錯，就只是永遠沒有東西可以分析。

**斜線指令沒出現**
全域同步最久要一小時。開發時把 `DEV_GUILD_ID` 設成你的伺服器 ID，會立刻生效。
另外確認邀請連結有勾 `applications.commands` scope。

**啟動時模型載入失敗**
看看 `vendor/MyGOChat/mygochat/models/model.safetensors` 的大小。如果只有一百多 bytes，
那它還是 LFS pointer：

```bash
git -C vendor/MyGOChat lfs pull
```

**`No module named 'torch'`**
你可能跑過 `SKIP_MODEL=1`。裝模型相依套件：

```bash
pip install -r requirements-model.txt
```

或是在 `.env` 設 `MYGO_ENGINE=random` 先用隨機模式。

**每次推論都跳 `torch.cuda.amp.autocast` 的警告**
上游的 `chatbot.py` 無條件呼叫了這個。在只有 CPU 的機器上它是 no-op，警告可以忽略。

**它只會對中文有反應**
對，模型只吃繁體中文。英文輸入基本上會得到亂七八糟的結果 —— 不過某種程度上這也算是特色。

---

## 備註與致謝

- 模型只支援繁體中文，上游回報準確率約 86%。
- 圖片是直接連 bahamut.com.tw，跟上游的做法一樣。
- 模型與圖片來自 [qaz45647/MyGOChat](https://github.com/qaz45647/MyGOChat)。
  串接細節寫在 `docs/MYGOCHAT.md`。
- 雖然 repo 叫 spammer，但拜託不要真的拿來洗頻。`cooldown_seconds`、`min_confidence`
  和預設為 0 的 `reply_chance` 都是有原因的 —— 一隻每則訊息都要回的 bot，下場是被成員
  靜音、被 Discord 限流。
