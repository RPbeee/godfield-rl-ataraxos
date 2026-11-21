# ランク戦キャンペーンの集計（`scripts/ranked_campaign.py`）

公式ランク戦（真剣タイマン、手順は `scripts/ranked_bot.md`）の台帳 `runs/ranked100/ledger.jsonl` を読み、
勝敗・決着不明・参加数・勝率の Wilson 区間・レート履歴を **1 つの JSON** にまとめる。

## 性質

- **読み取り専用**: 台帳を `open(..., 'rb')` で 1 回読むだけ。ロック（`ranked.lock`）を取らず、ファイルを作らず、
  台帳のフォルダに何も書かない。**ランクのループが台帳に追記している最中でも実行できる**
  （最後の改行までの完全な行だけを使い、書きかけの末尾は `ignored_tail_bytes` に数えて捨てる）。
- **匿名**: 出力は台帳にある数値・真偽値・固定 enum だけ。名前・ユーザー ID・部屋キー・URL・パス・受信データは
  台帳に無いので出ない。壊れた行・未知のキーの行は行番号だけ数え（`source.invalid_lines`）、中身は出さない。
- 実戦・専用プロフィール・ブラウザには触れない。`ranked_ledger` の純関数（検証・畳み込み・除外判定）だけを使い、
  `Ledger` クラス（ロックを取る）は使わない。

## 使い方

```sh
.venv/bin/python scripts/ranked_campaign.py --indent 2                   # 標準出力に JSON
.venv/bin/python scripts/ranked_campaign.py --out /tmp/campaign.json     # 同じ内容を 0600 で保存（台帳フォルダ内は拒否）
```

オプション: `--ledger`（既定 `runs/ranked100/ledger.jsonl`）、`--supplement`（既定
`measurements/ranked_game3_20261120/game3_rating.json`。数値だけの手書き補足）、`--supplement-game`（補足が
当てはまる番号、既定 3）、`--no-supplement`、`--indent`、`--out`。終了コード 0 = 出力、2 =
`error:ledger_missing` / `error:out_not_writable`。

## 出力の構成

| キー | 内容 |
|---|---|
| `source` | 読んだバイト数・行数・sha256（途中経過の比較用）、`ignored_tail_bytes`、`invalid_lines`（行番号）、`events`、`supplement` |
| `cap` | 数える上限 100、数える局数 `counted`、`remaining`、台帳の番号数、最大番号、台帳に記録された `budgets_recorded` |
| `preliminary_unapplied` | **先行 2 件（game 1・2）の別集計**。`games`・`count`・`stop_reasons`・`decisions`、`basis`（`pinned_ledger_events`＝固定した台帳イベントと完全一致で検証済み／`none`）。以降のどの数にも入れない |
| `counts` | `counted`、`participated`（実際に始まった局）、`completed`（結果が分かった局）、`wins`・`losses`・`draws`、`undecided_after_start`（始まったが結果不明）、`undecided_unmatched`（相手が付かず終了）、`in_progress`、`open_unmatched`、`closed_with_match` |
| `win_rate` | 95% Wilson 区間（`k`・`n`・`rate`・`lo`・`hi`）。`completed`（勝ち / 結果が分かった局、引分は勝ちに数えない）、`decided_only`（勝ち /（勝ち+負け））、`unknown_as_loss`・`unknown_as_win`（結果不明を全部負け／全部勝ちにした両端） |
| `by_opp_kind` | `human`/`bot`/`unknown` ごとの参加・勝敗・不明・進行中と、結果が分かった局の Wilson 区間 |
| `conditions` | 予約時の条件（`model`・`ckpt_sha8`・`code_rev`・`policy`・`search`・`dream`）ごとの番号。複数あれば警告 `mixed_conditions` |
| `games` | 局ごとの行。`klass`（`preliminary_unapplied`・`completed`・`undecided_after_start`・`undecided_unmatched`・`in_progress`・`open_unmatched`）、`counted`、結果・時間・判断数など、始まった局には `rating` |
| `rating` | 優先順位、局ごとの系列 `series`、自分の通算 `self`、相手の平均 `opponents`、**不一致の一覧 `mismatches`** |
| `warnings` | `game_in_progress`・`rating_mismatch`・`invalid_lines`・`unterminated_tail_ignored`・`mixed_conditions`・`game_numbers_not_contiguous`・`preliminary_games_not_verified`・`over_hard_limit` |
| `generated_at` | 実行時刻（unix 秒、CLI のみ） |

## レートの由来と優先順位

表示値は `1500 + 受信の生値`（公式クライアントの規則。`ranked_bot.md`「レートの生値と表示値」）。
台帳の旧書式（game 3・4 など、`rating_display_basis` の列が無い行）は生値しか持たないので、根拠のある値を
次の順に探す。**候補はすべて `candidates` に残し、値が食い違えば `agree=false` にして `mismatches` に挙げる**。

- 自分の開始（`self_before`）: ①押す前の公式画面（`rating_screen`）②新書式の表示列（`ledger_display`）
  ③手書き補足（`supplement`、game 3 の 1500）④生値+1500（`wire_offset`）⑤前の局の終了値（`prev_game_after`、連続性の確認）
- 自分の終了（`self_after`）: ①終局後の画面（`screen_after`）②**次の連続する局の押す前の公式画面**
  （`next_prematch_screen`、番号が +1 のもの）③新書式の表示列 ④補足（game 3 の 1516）⑤生値+1500
  ⑥次の局の開始の生値+1500（`next_wire_before`、最も弱い）
- 相手の開始・終了: ①新書式の表示列 ②生値+1500 ③補足

それぞれ `value`・`src`（採用した由来、無ければ `unknown`）・`candidates`・`agree` を持つ。局ごとに
`self_delta`・`opp_delta` を出し、`mismatches` には次の種類が入る: `value_disagreement`（候補どうしの食い違い）、
`not_zero_sum`（自分の増減 ≠ −相手の増減）、`self_gain_vs_delta`（生の gain と画面差の不一致）、
`outcome_vs_delta`（勝ちなのに下がった等）、`net_change`（全局が分かっているのに増減の合計と端点の差が違う）。

## 実データの見え方（2026-11-20 時点の台帳）

game 1・2 は `preliminary_unapplied`、game 3（勝ち）の自分は補足で 1500→1516・相手は 1532→1516、game 4（負け）は
自分が生値 16→1（表示 1516→1501、終了値は game 5 の押す前の画面 1501 とも一致）、相手 1639→1654。
game 5 以降は進行中（`game_in_progress`）。1 局単位の数字で、強さの根拠にはならない。

## テスト

`tests/python/test_ranked_campaign.py`（台帳の実数値を使った再現、Wilson の既知値、書きかけ末尾・壊れた行、
読み取り専用であること、補足の不一致、CLI）。先行 2 件の除外と上限は `tests/python/test_official_ranked.py`。
