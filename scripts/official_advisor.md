# 本家受信記録のオフライン助言

この入口はファイルだけを読み、通信・ブラウザ操作・モデル更新を行わない。

```sh
.venv/bin/cmake --build build -j 4
.venv/bin/python scripts/official_advisor.py \
  --input third_party/goodfield/data/official-v2/wire-game-updates-v73.json \
  --viewer-id 1 --checkpoint runs/m4-s-ppo-v1/checkpoint.pt --greedy --json
```

`--checkpoint` は `play.py` と同じ EMA loader。`--greedy` を省けば方策を
サンプルする。`--search --belief uniform`、`--belief b1`、学習済み belief の
path と `--fit` も使える。oracle は拒否する。CPU のみで実行する。
`--worlds`、`--candidates`、`--depth` で探索量を制御する。

入力は probe の capture prefix の本文ファイル、厳密 parser の出力 JSON、
`wire-game-updates` fixture、game の JSON 配列、game または `{game: ...}` の
JSONL。複数 document の prefix は `--document-sequence N` で一つを選ぶ。
同じ document の累積 prefix は重複 update を無視する。異なる対局を
一つの `OfficialSession` に混ぜず、対局ごとに作り直す。

## データの境界

1. `read_games` がローカル記録を読む。prefix は既存
   `parse_official_webchannel.py` の UTF-16 長・完全 frame の厳密解析を使う。
2. **`project_game` が wire を持てる最後の層**。新しい辞書を許可項目だけから作る。
   相手の未使用品については `used` の判定以外を転送せず、内容・ID・token・
   fake の有無・個数・空行の数を捨てる。相手の公開 used pane は face の列だけ。
   自分の偽装品の非公開 actual も捨て、face と偽装印と自己 item ID だけを残す。
   霧の相手 status は null。名前・userId・認証・room・transport 情報は含めない。
   相手向け gift/replaceItems の内容・個数も次の層へ渡さない。
3. `OfficialSession` が公開 use/target/damage/boost/exchange/advanceGF 等を順に読む。
   公開決定記録と前後の可視状態を作り、C++ の既存 `History::append` に渡す。
4. C++ `official_view` → 既存 `encode` → 全 `OBS_FIELDS`。
   `PublicSelection` → 既存 SelectionTree → 既存 gfrl `Selection` → prefix mask。
   相手の詰め物は**公開 used 品だけ**。未使用品を一枚も必要としない。
5. `OfficialDecision` は Actor と同じ numpy の観測/prefix 配列を持ち、`step` は
   選択だけ進める。一意の選択は VecEnv と同様に自動で進め、推論を行った回数を
   substep とする。選択を完了しても Match を実行しない。
6. `Presenter` で盤面・カード・選択を説明し、意味行動 JSON を返す。

`PublicSelection` は goodfield に追加した読み取り専用 API。公開 `MatchView` だけから
選択用の状態を構築し、`legal_kinds` と `legal_extensions` を既存の SelectionTree
へ委譲する。`step`、RNG、復元用の action journal は持たない。
`snapshot()` は Selection 用の合成状態で、実行可能な checkpoint ではない。
学習用 Match、VecEnv の既定、既存の snapshot/ABI/規則は変更していない。

## 履歴と決定数

`updateCount` は native の決定数として使わない。観測した useAttackItems /
useDefenseItems を公開決定にまとめる。通常補充は学習時の既定 Match と同様に
元の決定の効果に含め、fixture が再生用に使う external grant の追加決定とは区別する。

開始を見ていない場合は空の履歴から開始し、観測できた決定数を局所カウンタにする。
欠番・不明な効果・非公開実体に依存する効果・未対応のイベント時点の霧/守護神/災いの
変化などは `history_complete=false` と `partial_history_and_decision_count` で示す。
欠落区間を真手札や終点の真値で埋めない。途中参加時の観測は学習時の完全履歴と
同一とは主張しない。完全履歴を必須にする接続側は、このフラグで停止すること。
`public.last_decision` と `session.events` はこの公開決定の列を検査する入口。
複雑な効果の完全な履歴再構成は、今回の全 fixture について保証していない。

## 探索と未対応

主手番は公開情報だけの Match を作り、既存の SearchAgent と belief を使う。
相手の未使用品が空の構築元から belief の世界を作るので、受信した真在庫は使わない。
霧・偽装など既存 determinize が拒否する root は方策へフォールバックする。
応答/購入は合法選択と方策推論に対応するが、公開受信終点だけでは実行用 journal を
作れないので、**探索は方策へフォールバック**する。`search_status` に理由を返す。
これは `load_scene` の pending 制限を無効化する変更ではない。

`status=unsupported` は接続側が人へ知らせて停止すべき結果。
未知 schema/event、2 人以外/チーム、18 枠超過、不明な pending の行為者、
複数 pending frame、欠けた動的攻撃値、観測した launch のない redirect、
元の命中判定が不明な確率攻撃の redirect、混乱 main、Earth/Moon の一般 payload、
逆順/conflicting update などを理由コードで返す。

選択後にも `action_support` を検査する。94+232 の安全な限定例には
「相手に209が無い」という非公開条件が必要なため、真在庫を調べず unsupported。
偽装品を使う提案は、非公開 actual と一般 Dream 処理の適合を確認できないため
unsupported。可視 Earth/Moon がある場合も一般実行規約の差を明示して止める。
これらの出力にも検討用の意味行動は含まれるが、自動実行してはならない。
207 の Fate 起動は学習済み通常 Command の行動空間にないため、候補にならない
（捨てる/取引等の通常の役割は元の mask のまま）。全公式行動を網羅した mask ではなく、
**学習した gfrl の行動空間での mask**である。

一般の補充・病気・死亡・守護神 schedule や RNG の公式同等性は保証しない。
今回の API は効果をシミュレートして受信終点を推定するものではない。

## JSON の扱い

`action.cards` は選択順の `{model: 可視face, item_id: 受信した自己item.id}`。
`operations` は `kind → select_card* → finish_cards → target → allocate? → confirm`。
対象は公開 player ID、target=null は対象指定なし。UI の座標や DOM selector は含めない。
カード配列の添字やモデル ID を item ID の代用にしない。

`expected_update` / `expected_gf` / `viewer` で現在の受信状態と照合し、次段階では
item ID と face を現在の手札で再確認してからクリックへ変換する。
記録再生中の助言は実際には指していないので、後続の受信は助言への応答とは限らない。
raw、モデルエラー、認証、未知イベントの本文をエラー表示へ転記しない。
出力には自己手札が含まれるので、実対局では private な保存先を使うこと。

## 検証範囲

- 30 wire ファイル・79 game を fixture として使用。主手番43、応答待ち32
  （40.5%）、終局4。`load_scene` が受け付けない32応答待ちは読み取り専用の
  `PublicSelection` で扱い、実行用 journal を捏造しない。購入待ちは合成データでも検証。
- 76 game に既存 goodfield native 再生の対応あり。current board の 55 token は、
  **手札を対応付けると76/76一致**。native の外部補充決定数は受信 updateCount と
  異なるため、この比較では native の counter を明示して比較する。
- 74/76 は受信配列順のまま一致。v64/update4、v91/update2 は native 再構成の
  physical inventory 順が違う。face/used/disguised で自己 slot を対応付けると一致。
  この2件を受信順のテンソルの完全一致とは数えない。
- 対応なし3件: v82-hell/update1（UI 記録の初期だけ）、v93/update2 と v94/update3
  （確率 miss の record-only 終点）。native 遷移適合の件数に加えない。
- 対応する非終局72局面で native の mask と逐次 prefix の検証経路が一致。
  2件の在庫順差は mask の slot にも同じ対応付けを行う。
- 完全 start→end の v73/4状態と v87/2状態では、上記の補充決定を除いた
  学習時既定の counter に正規化した native 公開記録からの History と6/6一致。
- 全79入力で相手の未使用モデル/数/ID/fake と名前/userId を改変し、公開出力不変。
  75の viewer 決定で観測・mask・説明・意味行動も不変。霧の敵 status も別検査。
- goodfield に main/response/purchase の合法 prefix 一致テストを追加。

`gfrl_official_native_dump` は既存 `official_wire_trace_test.cpp` の実再生を
観測用 hook から取得するテスト専用実行物。native 真在庫を Python の変換側へ
渡さず、公開 view と encoder の出力だけを比較器へ渡す。

```sh
.venv/bin/cmake --build build -j 4
.venv/bin/ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q
# 親は既定で upstream tests を無効にするため、goodfield は別ビルドで全件検証
.venv/bin/cmake -S third_party/goodfield -B /tmp/gfrl-goodfield-offline \
  -G Ninja -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
.venv/bin/cmake --build /tmp/gfrl-goodfield-offline -j 4
.venv/bin/ctest --test-dir /tmp/gfrl-goodfield-offline --output-on-failure
```
