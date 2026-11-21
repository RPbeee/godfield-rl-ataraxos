# ランク戦キャンペーン運転（`scripts/ranked_drive.py`）

公式ランク戦（手順 `scripts/ranked_bot.md`）を、**同じ台帳 `runs/ranked100`・同じ名前・同じモデル**で、
**1 局ずつ・同時 1 本**、数える局が 100 になるまで安全に続ける再開可能なドライバ。戦闘ルーチン
（`ranked_bot.py`）は変更せず、1 局 1 プロセスの子として起動するだけ。実行は親が行う。

```sh
.venv/bin/python scripts/ranked_drive.py --playwright <playwright-core のパス> [--max-games N] \
    [--gap 60] [--accept-unresolved-through N] [--gate-ack <gate_id>] [ranked_bot の許可フラグ…]
```

状態はすべて台帳から読み直す（自前の状態ファイルなし）ので、止めてもそのまま再実行すれば続く。
**親の `ranked_loop.sh`（7 戦ループ）が終わってから**起動すること（動いていれば `ranked_process_running`
で何もせず終わる）。

## 流れ

1. **段階 1（既承認 10 件）**: 数える局が 10 未満の間、子に `--budget 12`（先行 2 番号 + 10）。
2. **10 件の関門**: 数える局が 10 になったら台帳を点検する（`ranked_campaign` の集計とレガシー補正込み）。
   - 最初の 10 局がすべて完走し、勝敗が勝ち／負けで、**自分・相手の前後の表示レート 4 値がすべて取れ**、
     勝ちなら自分の増減が正（相手は負）・負けなら逆、集計の不一致（`mismatches`）がなく、増減の合計が端点の差と一致
   - 数える局の予約条件（model・ckpt_sha8・policy・search・dream）が 1 種類で、`--checkpoint` の中身と一致
   - 異常があれば **100 へ進まず** `ev:"gate"` と `status:"gate_hold"`（終了コード 3）で理由コードと `gate_id` を出して止まる
3. **段階 2**: 関門を通ったら同じ台帳に `--budget 102`。数える局が 100 になったら `status:"done"`。

異常を確認した上で進めるときだけ、出力された `gate_id` をそのまま `--gate-ack` に渡す（異常の内容が
変わると id も変わるので、確認していない異常は通らない）。局が増えて 10 局目の根拠が変わった場合
（例: 11 局目の押す前の画面が 10 局目の終了値と食い違う）も、同じ関門で止まる。

## 安全の約束

- **干渉しない**: 各局の前に `ps` で `ranked_bot.py` / `ranked_loop.sh` / `ranked_drive.py` の実行を探し
  （自分と祖先は除く）、台帳の `ranked.lock` が掴まれていないか確かめる。どちらかなら `refused`（終了コード 2）。
  ドライバ自身の二重起動は `runs/ranked100.driver.lock` で防ぐ。ドライバは台帳に書かず、`ranked.lock` も持たない。
- **台帳の固定**: 先行 2 件（game 1・2）が固定イベントと完全一致する台帳だけを使う（`ledger_not_the_campaign_ledger`）。
  壊れた行・書きかけの末尾があれば `stopped`。
- **20 秒以上空ける**: 台帳の最後のイベントまたは子の終了から `--gap`（既定 60、20 未満は拒否）秒待ってから次の入場。
  起動直後の最初の 1 局にも適用する。
- **新規予約前の一時不良だけ再試行**: 子が台帳に何も書かずに終わり（= 予約前）、理由が
  `visible_state_timeout` / `login_not_accepted` / `ranked_lobby_missing` / `ranked_button_not_unique` のときだけ、
  30・60・120 秒待って**最大 3 回**。それ以外（`bot_prohibition_notice`、`ranked_game_in_progress_on_start` など）は即停止。
  使い切れば `transient_retries_exhausted`。
- **予約後は絶対に再試行しない**: 台帳に番号が増え、結果（勝・負・分）が記録されていなければ
  `unresolved_after_reserve`（局番号・始まったか・台帳の stop_reason・判断数）を出して停止。枠は消費済みのまま数える。
  再開時は結果のない数える局があると `unresolved_game` で拒否する。確認した上で
  `--accept-unresolved-through <その局番号>` を付けたときだけ続け（子に `--accept-unresolved` を渡す）、局は数えたまま残る。
  開きっぱなしの局（子が死んだ）も同じ扱い。
- **局ごとの軽い点検**: 終わった局の勝敗が自分のレート増減と食い違えば `outcome_contradicts_rating` で停止。
- **子へ渡せるフラグは許可リストだけ**: `--name --device --threads --match-wait --turn-wait --rating-wait
  --rating-screen-wait --timeout --settle-timeout --chance-delay --step --strict-events --watch-screen`。
  予算・状態ディレクトリ・チェックポイント・dream・resume・birth 関連は渡せない（`forwarded_arg_not_allowed`）。
  ドライバが `--budget`・`--state-dir`・`--checkpoint`・`--reenter` を付ける。**`--name` は以前と同じ値を親が渡すこと**
  （台帳に名前は残らないので照合できない。出力にも出さない）。
- **出力は固定コードと数値だけ**: すべて `clean()` を通り、小文字の固定コード・整数・真偽値・null 以外は拒否される。
  子の stdout は理由コードの取得にだけ使い、そのまま出さない。子の stderr は捨てる。
- 子が `--child-timeout`（既定 7200 秒）を超えたら SIGTERM（`ranked_bot` が記録して止まる）、120 秒で SIGKILL。
  ドライバへの SIGINT / SIGTERM は子へ SIGTERM を転送し、`interrupted` で終わる。

## 出力（1 行 1 JSON）

| `ev` | 内容 |
|---|---|
| `start` | 数える局数・残り・結果のない局番号・`code_revs`（予約に記録された code_rev の種類数）など |
| `gate` | `passed`・`acknowledged`・`gate_id`・`anomalies`（`g<局番号>_<内容>` の固定コード） |
| `launch` | 次の局番号・budget・何回目の試行か |
| `retry` | 一時不良の理由コード・何回目・待ち秒 |
| `game` | 終わった局: 勝敗・相手種別・判断数・自他の前後の表示レート・増減・通算の勝敗・`anomalies` |
| `end` | `status`（`done` / `max_games` / `stop_requested` / `stopped` / `gate_hold` / `interrupted` / `refused`）、`code`、`played`、数える局数・勝敗・残り |

終了コード: 0 = done / max_games / stop_requested、1 = stopped / interrupted、2 = refused（開始前の拒否）、3 = gate_hold。

穏やかに止めるには `touch runs/ranked100.stop`（実行中の局は最後まで打ち、次の入場の前に止まる。
ファイルがある間は起動しても何もしないので、再開前に削除する）。

## テスト

`tests/python/test_ranked_drive.py`: 子は **本物の `Ledger`**（予算・先行 2 件の除外・未確定の拒否が実コードで効く）を
一時ディレクトリに書く偽物で、時計も偽物。ブラウザ・ネットワーク・公式サイトには触れない。
全 100 局の通し、再開、関門（正常・欠落・勝敗矛盾・レガシー補正と補足）、一時不良の再試行と上限、予約後の不明と
`--accept-unresolved-through`、他プロセス・ロック・二重起動、台帳/チェックポイントの不一致、停止ファイル、
引数の許可リスト、出力に名前や子の出力が出ないこと、実サブプロセス経路（stderr 破棄・タイムアウト）を確認する。
