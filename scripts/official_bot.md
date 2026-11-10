# 本家の修行 UI bot（段階2）

ローカルのモデルから、既存 Node probe の可視 UI 命令を逐次作る。
本家への接続は通常実行を明示したときだけ。開発・テストには通信不要。
`room` の入口は分離済みだが未実装で、ブラウザ起動前に拒否する。

```sh
# 修行を開始し、最初の自分の決定の助言・予定クリックだけ記録して閉じる
.venv/bin/python scripts/official_bot.py \
  --mode training --checkpoint runs/play/m-2166.pt --greedy --games 1 \
  --playwright /absolute/path/to/node_modules/playwright-core \
  --out-dir /tmp/gfrl-training-dry-1 --dry-run

# 同じ設定で1局を自動操作
.venv/bin/python scripts/official_bot.py \
  --mode training --checkpoint runs/play/m-2166.pt --greedy --games 1 \
  --playwright /absolute/path/to/node_modules/playwright-core \
  --out-dir /tmp/gfrl-training-auto-1
```

`--dry-run` でも生誕・修行入室・開始のナビゲーションは行う。
対局のカード・対象・配分・送信のクリックは行わない。最初の助言で終了する。
通信しない検証には下記の `--replay` を使う。
`--name`（既定 GFRLBot）、`--search`、`--timeout`（受理待ち、既定20秒）、
`--settle-timeout`（画面・各クリックの反映待ち、既定30秒）、
`--on-stop save|abandon` も指定可能。探索は既存 advisor と同じく主手番だけで、
応答は方策へフォールバックする。複数局は `--games N`。
出力先の `moves.jsonl` は上書きしないので、毎回新しいディレクトリを使う。

受信は全 stream の不変 prefix を厳密 parser で解析する。同じ stream の
同じ SHA256 は解析し直さず、同じ updateCount の公開状態は一度だけ取り込む。
新しい prefix 内の game は updateCount 順。違う公開内容の重複、逆順、未知形式、
capture_error、整合しない bytes/hash は停止する。途中 frame は次回を待つ。
各局開始前の capture の frame 境界を基準にし、前局の受信を除く。
欠番は既存 Session と同じく部分履歴になる。名前は受信から記録せず、
修行の唯一の人間の ID を viewer とする。

`sell` は観測された `{action:"sell", item:{...}}` だけ許可する。
直前の公開 pending から買い手・売り手を決め、CP→MP→HP の支払と売り手の
CP増加を履歴に反映する。商品は買い手が viewer のときだけ可視 face を転送する。
買い手を特定できない場合も停止する。

クリック変換は1440×1000の観測済み配置に限定する。

操作前は0.5秒間隔で state と受信 capture を取り直し、GF・一意な自己名前帯・
未使用手札全体の画像と枚数・両者の資源・演出文言の不在を確認する。
応答時は相手の上欄の攻撃カードも照合する。待機中に updateCount が進めば
古い手順を中断し、新しい局面で助言し直す（JSONL に superseded と記録）。
カード選択・両替の各クリック後も同じ間隔で表示の反映を待ち、クリックは再送しない。
待機上限を超えたら直近の画面不一致理由で停止し、画面を保存する。

- catalog の category/imageName から model の画像 path を作る。手札の80×80画像を
  2段の手札領域だけで探し、自己 item ID/face/未使用状態と必要枚数を照合する。
  同じ face の偽装品があれば曖昧として停止する。
- probe に可視テキストの矩形を追加した。上部 y=55〜95 にある自分の名前帯から
  左350／右690の欄を決める。ヘッダや右側の status、フッタの名前を使わない。
- カードごとに自分の欄の画像 path と枚数を再確認する。GF と両者の可視資源も
  受信の公開状態と照合する。自分の欄は最初に空であることを要求する。
- `play`（攻撃・道具・奇跡・回復）、`sell`、`buy` はカードを順に選ぶ。
  `sell` は売るカード→商品の2枚、`buy` は買うカードの1枚。
  送信直前に上部の名前帯を照合し、自分と反対側に相手の名前帯があれば
  助言も相手対象であることを要求する。自分の名前帯1本なら対象なし／自分のみ
  許可する。対象をクリックして補正せず、不一致なら画面と state を保存して停止する。
- `pray` はカードを選ばず自分の欄をクリックする。下部の「祈る」は押さない。
  捨てる、購入 YES/NO、その他の未観測種類、霧・偽装の画面、5枚以上の
  上欄表示は停止する。空応答は対応済み。
- 両替は MP/CP の差分を ±10/±1 に分け、HPを自動計算する。中間値が範囲を
  越えないよう減額を先に行う。各クリック後の3つ目の HP/MP/¥ 表示を照合する。
  35/35/0では所持金 -10×2 → MP +10×2/+1×5 となり、実測と同じ配分。
- 送信は自分の上欄の中央 x、y=193 の隙間。両替だけは観測済み左欄 (350,250)。
  下部の攻撃／防御／許す表示は押さない。送信後は capture で updateCount の増加を
  待ち、同じ手を再送しない。更新増加は受理の最低条件で、結果の効果を推測しない。
  実行内で各種類が初めて受理された行だけ `first_success_kind` に種類名を記録する。

未対応局面、画面不一致、受理タイムアウト、probe 異常で停止する。
既定は `stopped-state.json`（本文・画像・テキスト矩形など probe の state）と
`stopped-screenshot.png` を保存してブラウザを閉じる。`--on-stop abandon` では保存後に
観測済みの戻る → 戦いをあきらめる → メニュー復帰を試みる。probe が異常終了した
場合は画像保存や退出を保証できない。終局時は dead の公開状態から勝敗集計し、
「戦いを終わる」で設定画面への復帰を確認する。

保存ディレクトリは0700、JSONL・停止 state・停止画像は0600。
手の JSONL は数値IDの局面要約、名前を含まない助言、クリック、受理結果だけ。
raw受信・本文・認証は JSONL に含めない。probe が必要とする受信 prefix と通常の
画面データは private な一時ディレクトリに保存し、close後に削除する。
停止 state と画像には表示名や自己手札が含まれるため private に扱うこと。

## オフライン検証

```sh
.venv/bin/python scripts/official_bot.py \
  --checkpoint runs/play/m-2166.pt --greedy --dry-run \
  --replay /absolute/path/to/live1 --out-dir /tmp/gfrl-live1-replay-1
.venv/bin/python -m pytest -q tests/python/test_official_bot.py
```

`--replay` は live1 の resp.jsonl と prefix を読む専用アダプタ。
Nodeもブラウザも起動せず、元資料は変更しない。5つの自分の決定すべてについて
モデルの助言と予定クリックを表示する。記録の後続更新を助言の受理と認定しない。
古い記録には名前帯の矩形がないため、live1の画面で確認した左／右の位置を明示して
補う。実行中の bot はこの補完を使わず、probe の実測テキスト矩形を要求する。

テスト fixture は6更新の匿名化した最小 game literal で、相手の未使用在庫を除いた。
合成 probe は生誕→設定確認→5決定→複数 stream の受理→終局→設定復帰を通す。
不一致・タイムアウト・JSONL subprocess・隠れ在庫変更時の不変性も検証する。
新しい本家接続での成功を保証するものではなく、未観測の手に達したら停止する。
