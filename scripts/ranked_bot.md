# 公式ランク戦（真剣タイマン）100局の実行手順

**ユーザーの明示承認つきの唯一の例外**（公開対戦禁止の例外）。運営のメール許可取得済み・追加条件なし。
探索なしの m-xit1 段階別 greedy、公式の真剣タイマンで **1局ずつ、最初の10局を含め総計100局まで**。
実戦は親（人間側のセッション）が起動する。実装側・テストは実機のブラウザや専用プロフィールを起動しない。
pytest も親が実行する（`tests/python/test_official_ranked.py`）。連続運転は `scripts/ranked_drive.md`、
台帳の集計は `scripts/ranked_campaign.md`。経緯と戦績は `AGENTS.md` §5.22。

## 守ること（コードで強制）

- **保存・出力してよいのは数値・真偽値・固定コードだけ**（`python/gfrl/ranked_ledger.py` の allowlist。未知のキー・
  自由文字列は書込み前に拒否）。名前・ユーザーID・合言葉・部屋キー・URL・受信の生データ・画像・チャット・console は
  保存も表示もしない。`moves.jsonl`（Journal）は使わず、停止時のスクリーンショット・stopped-state も作らない。
  stdout は結果 JSON 1行だけ、stderr は捨てる。
- **受信の生データはメモリだけで扱う**。probe（`scripts/official_probe.mjs`、`GFRL_MEMORY_ONLY=1` で third_party の
  probe をソース上でパッチ。third_party 自体は不変）がストリームをメモリに持ち、stdout のパイプで base64 のまま
  Python に渡す（`Probe(memory=True)`）。生の prefix・capture のメタデータ・スクリーンショットはディスクに作らない。
  `--out-dir` は作らず、パッチ後のソースにファイル書込み API が残れば起動しない。
- ブラウザは専用の永続プロフィール `<state-dir>/profile`（0700、既定は `runs/ranked100/`。`runs/` は git 管理外で、
  git 管理外でない場所を指すと起動しない）だけを使う。同じアカウント＝同じレート。個人アカウントは使わない。
  HTTP/メディアのディスクキャッシュ、ログ、クラッシュダンプ、HAR、動画、トレースは無効。
- 通常のブラウザ UI だけを使う。非公開 API・他人の手札は使わない。
- 台帳 `ledger.jsonl`（追記専用・fsync）は研究番号だけで識別する。**「2人個人戦」を押す前に枠を予約**する。
  決着不明（停止・中断・時間切れ・プロセス死）の枠は戻さず、`flock` で同時 1 プロセス。
  前の局が未決（`previous_game_unresolved`）なら次の予約を拒否し、親が確認して `--accept-unresolved` を付けたときだけ進む。
- 「BOT 禁止」の表示を検出したら応募前でも待機中でも止まり `bot_prohibition_notice` を返す。回避しない。**止まったら親へ報告**。
- 終局後の退出: 公式の終局確認と台帳の finish 行（fsync 済み）の**後にだけ**「戦いを終わる」を最大 1 回押す。
  対戦中・あきらめる等は押さない（下の「終局後の退出」）。

## 数える上限 100 と先行 2 件

- 数える上限 `HARD_LIMIT = 100`。応募した局・決着不明の局・進行中の局はすべて数える。`reserve` は
  「台帳の番号の総数 ≥ `--budget`」または「数える局数 ≥ 100」で拒否する（`budget_exhausted`）。
- 先行 2 件（game 1・2）は「戦う」を押していない未応募・相手なし・判断 0 の `match_wait_timeout` で、**実対局は 0**。
  台帳の該当イベントが `ranked_ledger.PRELIMINARY_UNAPPLIED` の固定値と**完全一致**したときだけ数えない
  （時刻・件数・番号・フォルダが違えば数える側）。台帳には書き足さず、`schema` は 1 のまま。
- `--budget` は「台帳が通算で持てる番号の総数」。先行 2 件が検証済みの台帳だけ 102 まで許し、実戦 100 局には
  `--budget 102`。それ以外で 100 を超えると `invalid_budget`。台帳番号と実対局数は別物として扱う。
- `--status` と結果 JSON の `ledger` に `counted`・`exempt_unapplied`・`unresolved_counted`・`consumed`・`remaining` が出る。

## 手順

```sh
cd ~/godfield-rl-ataraxos
PW=<playwright-core のパス>
.venv/bin/python scripts/ranked_bot.py --status                                  # 台帳の確認（ブラウザなし）
.venv/bin/python scripts/ranked_bot.py --discover --playwright "$PW"            # ロビー確認（予約も応募もしない）
.venv/bin/python scripts/ranked_bot.py --budget <N> --step 戦う --reenter --playwright "$PW"   # 1 局
scripts/ranked_loop.sh <回数> <budget> "$PW" --step 戦う --reenter              # 1 局ずつ連続（0 以外で中断）
```

- 1 回の起動 = 1 局。結果 JSON の `status` は `complete`（終局を受信）か `stopped`（理由コード付き）。
- 新しい専用プロフィールの最初だけ `--discover --allow-birth` でアカウントを実装の既定名で作る（以後は付けない）。
- 同じプロフィールでは、ページを開くたびに出る「生誕する」画面を `--reenter` が同じ既定名で押して入り直す
  （marker と保存ログインが条件。保存ログインが無ければ `reenter_login_missing` で止まり、新アカウントは作らない）。
- 標準の経路は「2人個人戦」→（画面の自分のレート数字が読めてから）「戦う」→ 対局 → 「戦いを終わる」。
  「戦う」は `--step 戦う` で押す。`--step` の許可語は「戦う」「戦いを始める」「開始」「OK」だけで、既定は何も押さない。
  各語は 1 局で最大 1 回、待機中で、その語が 1 つだけ見え、押す前のロビーに無かったときだけ。台帳には触れず、
  「2人個人戦」は二度押さない。
- オプション: `--name`（省略時は既定名。ログに出さない）、`--match-wait`（マッチ待ち、既定900秒）、`--turn-wait`（相手の手待ち、
  既定900秒）、`--rating-wait`（終局後のレート更新待ち、既定10秒）、`--rating-screen-wait`（押す前のレート数字待ち、既定30秒）、
  `--leave-wait`（退出ボタン待ち、既定5秒）、`--resume`、`--accept-unresolved`、`--strict-events`、`--dream`、
  `--checkpoint`（既定 `runs/play/m-xit1.pt`）。
- 停止時の結果 JSON と `--discover` には、固定語彙の `screen` と `steps` が出る（本文・名前・URL は出ない）。

## 台帳に残る項目（すべて数値か固定 enum）

`game`、予約・開始・終了時刻（unix 秒）と `duration_s`、`seat_id`、`first_mover`（self / opponent）、`outcome`（win / loss / draw /
unknown）、`terminal`（completed / stopped / unresolved）、`stop_reason`、`decisions`、`updates`、`gf_final`、`end_gf`、
`opp_kind`（human / bot / unknown）、`resumes`、条件（`model`、`ckpt_sha8`、`code_rev`、`policy=greedy`、`search=false`、`dream`）、
レート（次節。表示値・生値・照合の 3 系統で、混ぜない）。

敗戦・時間切れ・中断を除外しない。終局を受信できなかった局は `outcome=unknown`、枠は消費のまま。
こちらが対局中に止まると相手には切断として見える。

## レート（表示値と生値）

**受信の `rating` / `ratingGain` は内部値で、表示 = 1500 + 内部値**（親が取得した公式公開 JS で確認。sha256
`ff77fdd0f7758f6c3c87121bba3902ec8e3d848baa8f04ba7064d6ef35fdfd2b`）。キーの欠損は 0。生の値を表示レートとして読まない。

- 生値: `rating_{self,opp}_first/last/gain/before/after` と由来 `rating_origin`（`wire_consistent` / `derived_from_gain` /
  `wire_before_only` / `unknown`）。受信のまま、換算しない。根拠のない値は null。
- 表示値: `rating_{self,opp}_display_before/after`（= 1500 + 生値、`WIRE_DISPLAY_OFFSET`）と由来 `…_src`
  （`wire` / `wire_absent_zero` / `screen` / `unknown`）、`rating_display_basis`（`offset_verified` / `unconverted`）。
  欠損 = 0 とみなすのは「既知の席の行でキーが欠損」のときだけ（rating は全観測で欠損、gain は正式終局の行で欠損）。
  null・文字列、欠損と整数の混在、未観測の席は `unknown`。自分の終了値は、読めれば終局後の画面を wire より優先する。
- 画面値: `rating_screen`（「戦う」を押す前）、`rating_screen_after`・`rating_screen_gain`（終局後の結果画面）。数字のみ。
- 照合: `rating_screen_check`（`consistent` / `inconsistent` / `delta_only` / `unavailable`）、`rating_gain_cross`
  （`zero_sum` / `not_zero_sum` / `unknown`）。`inconsistent` や `not_zero_sum` は止めて親が確認する。

**局ごとの条件（コードで強制）**: 「戦う」を押す前に、画面の自分のレート数字が同じ値で連続 2 回読めるまで待つ
（`--rating-screen-wait`、読めなければ押さず `rating_screen_not_loaded` で停止。枠は消費のまま）。終局後は
`--rating-wait` の間、結果画面の表示レートと符号つき増減を許可語彙と数字だけで読み、読めなければ null（局は止まらない）。

**実機で確認済み**（台帳 game 3〜9）: 各局の押す前の画面レートは、直前の局の終局後の値と毎回一致する
（1516 → 1501 → 1484 → 1500 → 1483 → 1501 → 1517）。局ごとにプロセスを起動し直しても同じアカウントのままレートが継続する。
自分の席の生値は game 3・7 で欠損（欠損 = 0 → 開始 1500）、他は存在した。台帳の game 3 行は追記専用で、自分の生値が全 null のまま。
補足は `measurements/ranked_game3_20261120/`（git 管理外）。
**未確認**: 結果画面の文言（`rating_screen_check` は全局 `unavailable`、`rating_gain_cross` は `unknown`）。
画面の語彙は `--discover` の `screen` で確かめてから形式を確定する。

## 終局後の退出（「戦いを終わる」、ranked のみ）

`RankedSession.leave_end_screen`。前提は公式の終局確認と台帳 finish 行の fsync が済んでいること（未了なら `not_finished`）。
`--leave-wait` の間、「戦いを終わる」がちょうど 1 回見えるのを待ち（無ければ `button_absent`、複数なら `button_ambiguous`）、
1 回だけ押し（再試行しない）、本文から「戦いを終わる」と `G.F.` が消えたら `left`、確認できなければ `left_unconfirmed`、
例外は `leave_failed`。結果 JSON の `end_screen` に出るだけで台帳は変わらない。room / training は従来どおり。
目的は、終局画面を離れずにブラウザを閉じると次局の入場が `visible_state_timeout` になりやすいため。
再起動直後の `visible_state_timeout` は 15〜30 秒待つと復旧する（`ranked_drive.py` が 25 秒空けて再試行）。
退出で減るかは未確認（`end_screen` が `left` 以外で続くなら `--leave-wait` と遷移後の画面を親が確認する）。

## 再接続（`--resume`）

プロセスが死ぬと台帳に未決の枠が残り、既定では次回起動時に `unresolved` として閉じる（枠は消費のまま）。
ページが対局画面のまま残っていて未決の局があるときだけ `--resume` で同じ番号に再接続できる（枠は増えず、
「2人個人戦」は押さない）。再接続前の履歴は受信に無く、`decisions` / `updates` は再接続後の分だけ（`resumes` > 0 の印）。
ブラウザ再読込で公式が対局を復元するかは未確認なので、基本は使わない。
