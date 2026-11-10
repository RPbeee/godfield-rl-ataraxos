# 本家の修行・合言葉の部屋 UI bot

ローカルのモデルから、既存 Node probe の可視 UI 命令を逐次作る。
本家への接続は通常実行を明示したときだけ。開発・テストには通信不要。

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
`--name`（既定 GFRLBot）、`--search`、`--timeout`（ログイン・送信の受理待ち、既定20秒）、
`--settle-timeout`（画面・各クリックの反映待ち、既定30秒）、
`--on-stop save|abandon` も指定可能。探索は既存 advisor と同じく主手番だけで、
応答は方策へフォールバックする。複数局は `--games N`。
出力先の `moves.jsonl` は上書きしないので、毎回新しいディレクトリを使う。

受信は全 stream の不変 prefix を解析する。bot は events だけ寛容に取り込み、
未知のイベントを内容のない `unknownEvent` に置き換える。履歴は部分記録となり、
安全な key 名と action 値だけを助言の警告・`moves.jsonl`・標準エラーに残す。
`--strict-events` で未知のイベントでも停止する。parser 自体の既定は厳密のまま。
players・attacks・その他の盤面は常に厳密に検査する。同じ stream の
同じ SHA256 は解析し直さず、同じ updateCount の公開状態は一度だけ取り込む。
新しい prefix 内の game は updateCount 順。違う公開内容の重複、逆順、盤面の未知形式、
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
応答時は相手の上欄の攻撃カードも照合する。最新の受信に hit/miss があれば、
`--chance-delay`（既定3秒、0も可）だけ待ってから画面確認とクリックへ進む。
この待ちも受信を確認し、同じ hit/miss が残っていても時間経過後は進む。待機中に updateCount が進めば
古い手順を中断し、新しい局面で助言し直す（JSONL に superseded と記録）。
カード選択・両替の各クリック後も同じ間隔で表示の反映を待つ。
カード選択だけは待機上限後も選択欄がクリック前と同じで、手札・攻撃欄・名前帯・矢印・
GF・資源に変化がなく、受信も進んでいなければ、そのカードを1回だけ再クリックする。
成功時は `selection_retries` にカードの0始まりの選択番号を記録する。
両替の増減操作は再送しない。再試行後も反映されなければ直近の不一致理由で停止し、画面を保存する。

- catalog の category/imageName から model の画像 path を作る。手札の80×80画像を
  2段の手札領域だけで探し、自己 item ID/face/未使用状態と必要枚数を照合する。
  同じ face の偽装品があれば曖昧として停止する。
- probe に可視テキストの矩形を追加した。上部 y=55〜95 にある自分の名前帯から
  左350／右690の欄を決める。ヘッダや右側の status、フッタの名前を使わない。
- カードごとに自分の欄の画像 path と枚数を再確認する。GF と両者の可視資源も
  受信の公開状態と照合する。自分の欄は最初に空であることを要求する。
  選択枚数は goodfield の hand_limit まで許可し、自分の欄の x 範囲かつ y<480 の
  80×80画像を数える。縦の間隔や段数には依存しない。
- `play`（攻撃・道具・奇跡・回復）、`sell`、`buy` はカードを順に選ぶ。
  `sell` は売るカード→商品の2枚、`buy` は買うカードの1枚。
  送信直前に上部の名前帯を照合し、自分と反対側に相手の名前帯があれば
  助言が相手対象または target None であることを要求する。自分の名前帯1本なら
  対象なし／自分のみ許可する。2人戦の target None は両方の表示を受け入れるが、
  対象側に自分の帯が出たら停止する。どちらの表示も確定は自分の欄の同じ座標を押す。既定の対象と助言が違う場合は、右の状態欄で対象 player の名前帯の
  中央を1回クリックする。上部に自分→対象の2本の帯と矢印が出ることを確認し、
  反映されなければ `target_switch_failed` で停止する。別の場所へのクリックは試さない。
  既定が相手の操作（goodfield の `targets_enemy_by_default` が真のカードを基底とする
  `play`・`sell`・`buy`）を自分へ向ける手は、`action_support()` が
  `self_target_of_enemy_default_unobserved` として未対応にする。武器・通常攻撃・
  売る・買う・CPを奪う・状態付与・除去などを含む。命中率付きカードなどの例外も
  goodfield の判定に従う。bot は同じ公開局面で対象選択から自分だけを除いて
  作り直し、カードと相手対象を残す。`mask_exclusions` には理由と `target: self` を記録する。
  auto37 の `moves.jsonl` 43行目（update 43 / GF 59）は、5「買う」を自分へ向け、
  状態欄を1回クリックしても上部が自分→相手のままで `target_switch_failed` となった記録。
  逆方向（既定が自分の道具を相手へ向ける）の切り替えは維持する。
  手元の auto1〜37・room 系 `moves.jsonl` には逆方向の切り替えが受理された実例は
  見つからなかった。198・201 の切り替え成功／失敗は合成 probe テストで検証している。
- `pray` はカードを選ばず自分の欄をクリックする。下部の「祈る」は押さない。
  未観測の `discard`・`sacrifice` は `ui_kind_unobserved:<kind>` として
  kind の mask から除き、同じ greedy/sample 設定で助言を作り直す。
  除外は `moves.jsonl` の `mask_exclusions` に理由と `kind` を記録する。
  合法な選択が残らない場合は `selection_exhausted` とする。Dream のみ、下記の
  最後の手段で偽装カードの除外を緩める。
  UI に直接渡された未観測種類は `unobserved_ui_kind` で停止し、detail に kind を残す。
  偽装カードの未検証理由と個体 ID の除外は捨てる手にも維持する。空応答は対応済み。
- 自分が霧のときは、右の状態欄で自分の名前がある段と反対の段に、名前・数字の
  無い紫の帯があることを確認する。上段／下段のどちらの並びにも対応する。
  応答の局面で `--settle-timeout` まで待っても上部に自分の帯が無ければ、
  `fog_response_ui_unobserved` で停止する。
- 両替は MP/CP の差分を ±10/±1 に分け、HPを自動計算する。中間値が範囲を
  越えないよう減額を先に行う。各クリック後の3つ目の HP/MP/¥ 表示を照合する。
  35/35/0では所持金 -10×2 → MP +10×2/+1×5 となり、実測と同じ配分。
- 送信は自分の上欄の中央 x、y=193 の隙間。両替だけは観測済み左欄 (350,250)。
  選択後のカード画像と重なる場合は、欄の左右端近くの画像外の点を探す。
  確定点が見つからなければ `confirmation_point_unavailable` で停止する。
  下部の攻撃／防御／許す表示は押さない。送信後は capture で updateCount の増加を
  待ち、同じ手を再送しない。更新増加は受理の最低条件で、結果の効果を推測しない。
  実行内で各種類が初めて受理された行だけ `first_success_kind` に種類名を記録する。

未対応局面、画面不一致、受理タイムアウト、probe 異常で停止する。
既定は `stopped-state.json`（本文・画像・テキスト矩形など probe の state）と
`stopped-screenshot.png` を保存してブラウザを閉じる。
`fog_response_ui_unobserved` のときは、その後5秒おきに2回、
`stopped-state-2.json` / `stopped-screenshot-2.png` と `-3` の組も保存する。
`--on-stop abandon` では追加観測を含む保存後に
観測済みの戻る → 戦いをあきらめる → メニュー復帰を試みる。probe が異常終了した
場合は画像保存や退出を保証できない。終局時は dead の公開状態から勝敗集計し、
「戦いを終わる」で設定画面への復帰を確認する。

保存ディレクトリは0700、JSONL・停止 state・停止画像は0600。
手の JSONL は数値IDの局面要約、名前を含まない助言、クリック、受理結果だけ。
raw受信・本文・認証は JSONL に含めない。probe が必要とする受信 prefix と通常の
画面データは private な一時ディレクトリに保存し、close後に削除する。
停止 state と画像には表示名や自己手札が含まれるため private に扱うこと。

停止時の `moves.jsonl` は `reason` と安全な `detail` を持つ1行を記録し、
標準エラーにも同じ2項目を1行の JSON で出す。停止行には助言本文をコピーしない。
「生誕する」の後、`--timeout` を過ぎても名前入力画面なら `login_not_accepted` で停止し、
標準エラーに「名前が使用中の可能性。別の名前で」を追加する。名前そのものは出さない。

- 受信 parser の拒否は例外クラス・固定コード・`parser` の詳細。
  未知の key は安全な綴りだけ最大16個、その他は件数だけ。場所は
  `game.attacks[].buyingItemModelId` のように添字を省く。
  文字列の値を出せるのは `action` / `curse` / `curses` / `guardian` の短い英数字識別子だけ。
  名前と秘匿 key の配下は `?` にし、その他の値は型と長さだけにする。
  `buyingItemModelId` の受理条件は変更していない。
- 公開状態の未対応は action、phase、公開 pending の model、条件の成否など。
  相手の未使用在庫は診断にも渡さない。
- カードの不一致は `expected` / `actual` の公開画像 path 別枚数と、
  `clicks` の `path` / `x` / `y` / `appeared`。未知の画像 path は `?` にする。
  資源・両替は期待値と読み取った数値、名前帯は本数と x 座標だけ。
- 想定外の例外はクラス名と末尾フレームのファイル名・行番号だけ。
  受信境界では goodfield/gfrl 内の末尾フレームを使う。例外メッセージ、
  生受信、認証、表示名、画面本文は `detail` に含めない。

## 合言葉の部屋（隠れ乱闘・2人個人戦）

```sh
# 合言葉はリポジトリ外の0600のファイルに用意する。
# 最初の自分の決定だけ助言し、対局の手は送信せず閉じる
.venv/bin/python scripts/official_bot.py \
  --mode room --password-file /tmp/gfrl-room-password --name GFRLBot \
  --checkpoint runs/play/m-2166.pt --greedy --games 1 \
  --playwright /absolute/path/to/node_modules/playwright-core \
  --out-dir /tmp/gfrl-room-dry-1 --dry-run

# 人が開始し、1局を自動で指す
.venv/bin/python scripts/official_bot.py \
  --mode room --password-file /tmp/gfrl-room-password --name GFRLBot \
  --checkpoint runs/play/m-2166.pt --greedy --games 1 \
  --playwright /absolute/path/to/node_modules/playwright-core \
  --out-dir /tmp/gfrl-room-auto-1
```

`--password '<合言葉>'` でも指定できるが、シェル履歴・プロセス引数には注意する。
`--password-file` と同時指定はできない。ファイルは通常ファイル・権限0600が必須。
合言葉・名前は moves.jsonl、標準出力、標準エラーに書かない。

bot はロビーの「隠れ乱闘」→合言葉入力→「唱える」で入室し、
「終末の時」を「なし」に変更して反映を待つ。中央の個人戦参加者欄に
自分がいない場合だけ左の丸 (400,173) を1回押し、自分が1回だけ出ることを確認する。
ヘッダや右下の在室者一覧は参加者に数えない。チーム戦の枠は使わない。
入室直後に対局・終局画面なら、受信 game の players に自分の名前がある場合は
`room_previous_game_in_progress` で停止する。名前がなければ観戦として、
「戦いを終わる」が見えたら1回押し、対局中なら `--room-wait` の範囲で部屋への復帰を待つ。

人は同じ合言葉で「隠れ乱闘」に入り、「個人戦」の丸を押して参加し、
「終末の時：なし」を維持して「戦いを始める」を押す。
`--room-start human` が既定。`--room-start bot` なら中央の参加者が
自分ともう1人のちょうど2人になったとき、bot が開始を押す。
入室後の開始待ちは `--room-wait`（既定1800秒）、対局中の相手の手番待ちは
`--turn-wait`（既定900秒）。`--timeout` はログインと自分の送信の受理待ち、
`--settle-timeout` は画面・設定の反映待ちとして使う。

受信 game の players.name が `--name` と厳密に1人だけ一致する席を viewer とする。
一致0人・複数人、2人戦でない受信、チーム戦、終末あり、表示と受信の不一致は停止する。
参加者3人以上、中央に同じ自分の名前が複数、参加の確認失敗、終末なしにできない／
開始待ち中に変更された、待ち時間超過でも開始・送信を進めず停止する。

指定局数の終局を受信したら、勝敗にかかわらず `status: complete`・exit 0 で終了する。
最終局では結果表示・部屋復帰・ロビー遷移を待たずに close する。
次局があるときは「戦いを終わる」または既に部屋へ戻った表示を待ち、
終末なし・参加を再確認して次局の開始を待つ。
停止時と dry-run の最初の助言後は、その場で close する。
対局中から部屋への未観測の放棄操作は行わない。room の `--on-stop abandon` は引数エラー。
停止時は画像をprivateに保存し、停止 state の本文・入力値・テキストは伏せて矩形だけ残す。
合言葉入力画面で止まった場合は、入力を伏せてから画像を保存する。
入力の置換を確認できない場合は、合言葉を残さないため画像保存も中止する。
画像自体には表示名が映りうるため、共有・リポジトリへの追加はしない。

SELL・BUY の受信 atk は威力として使わず、商品が武器でも防具を合法応答にしない。
助言の board は「威力」の代わりに「取引」と表示する。atk の値の意味は未確定。

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

### 2r 実機停止の回帰修正

- 霧の帯は半透明なので、内側の RGB 範囲と R/G 差で判定する。左側の丸みを避けた x=870–1194、上下 5px を除く領域を検査する。帯内に名前・数字がないことは引き続き `texts` で確認する。回帰テストは実測色に基づく合成 PNG のみを使う。
- Earth/Moon の所持自体は決定を止めない。公開の guardian identity は観測に入るが、将来の一般神器・奇跡の抽選は現在の合法手を決める入力ではない。`earth_moon_general_execution` は助言の警告に残す。受信済み攻撃は v33 typed payload の native `guardian_payload_legal` でモデル列の domain を検査し、対応範囲だけ通常の公開応答カーソルへ渡す。動的威力・redirect 等の既存停止条件は維持する。
- native の Earth120/Moon26 の各単品と Earth の複合列から作った応答について、公開観測・mask を照合する。205/206 の公開 amount も native と同じ 3/2 にする。守護神なし・Earth・Moon・Mars の main/通常応答では、native と公開選択木の全接頭辞を比較する。
- Dream の選択は見かけと個体 ID に基づくが、goodfield の `docs/official-execution-subset.md` の v28/v32 は互換クラス外の実体不適合入力について公式 server の受理規約を確定していない。discard でも actual を reveal する共有消費経路を使うため、偽装カードはまず `dream_actual_execution_not_verified_from_public_faces` として除外する。合法な祈り・空応答を含む安全な候補が尽きた場合だけ、2x の最後の手段を使う。
- auto39 の最終停止行は GF38/main。除外 5 件の後、最後の除外で公開選択が尽きると、その除外を戻して再抽選していた。これをやめ、最後の除外も `mask_exclusions` に残し、`selection_exhausted` で停止する。安全な候補がある場合は除外後の mask で選び直す。同一 face の先頭コピーを除外したとき、canonical な重複省略で後続コピーまで選べなくならないよう、除外中だけ全順序の選択木を使う。学習・収集・評価用の選択木は変更しない。

### 2s 実機停止の調査（オフライン）

霧の扇97は台帳では水属性・50%攻3・被害時Fogであり、Fogは攻撃属性ではない。
[属性規約](../third_party/goodfield/docs/official-current-rules.md#光闇と属性合成)の火／水の対属性と
`official_rules.cpp` の `attribute_defense_allowed` / `can_append_response` に従い、
火属性DEF3の126は合法。合成応答のnative maskでも選択できることを回帰検証する。
停止画像は手札126が残り、自己選択欄が空の状態だった。保存されたmovesには
更新45/GF52の操作受理→更新46/GF53の選択不一致があるが、時刻付き受信本文・
当該hit/miss列は保存されていない。probeの一時受信は終了時に削除されるため、
命中演出とクリックの厳密な時刻関係は確定できない。合法性を変えず、画面と受信の
再確認で安全に限定した再試行を行い、なお不一致なら停止する。goodfieldは変更しない。

部屋の負け側はfinished後、停止画像では既に部屋画面に戻っていた。従来は
「戦いを終わる」を必須で待っていたためvisible_state_timeoutになった。
最終局は受信の終局で完了とし、次局がある場合も部屋への直接復帰を認める。


### 2x 実機停止の修正（オフライン）

- 2人戦の target None はカードの種類に依存せず、自分の帯1本と相手の対象帯ありの
  両方を許可する。対象側が自分なら停止する。match1 の 232 と auto50 の 104 を
  合成 state で検証し、両表示の確定操作が同じ自分の欄のクリックであることも確認する。
- Dream は元の公開手札を保持したまま mask を作り直し、偽装カードと未観測の kind を
  除外する。合法な祈り・空応答が残るなら、それを優先する。武器を選択から除外しても、
  手札に武器があるため禁止される祈りを新たに合法にはしない。
- 安全な mask が尽きた場合に限り、偽装カードの除外だけを戻す。その他のカード・対象の
  除外は維持し、discard・sacrifice は常に禁止する。Dream の未検証理由は警告に下げ、
  `moves.jsonl` の行と助言に `dream_last_resort: true`、助言の `warnings` に
  `dream_actual_execution_not_verified_from_public_faces` を残す。停止時にも固定の警告を残す。
  UI はこの場合だけ見えている面をクリックする。
- 受理は従来通り受信の updateCount の前進を必須とし、受理されなければ停止する。
  `OfficialSession.ingest()` は受信 game の公開状態を盤面に投影するため、使用結果の実体が
  偽装の面と異なっても、選択した面から盤面を推測して上書きしない。合成受信と probe で
  実体の違う使用結果・正常受理・受理タイムアウトを検証する。
