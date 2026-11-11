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
path と `--fit` も使える。oracle は拒否する。belief は `--search` のときだけロードする。
`--device cpu|mps`（既定 cpu）、`--threads`（既定1）を指定できる。
`--worlds`（16）、`--candidates`（8）、`--candidate-samples`（64）、`--depth`（3）、
`--alpha`（0.002）、`--beta`（0.02）、`--leaf-view actor|searcher`（actor）、
`--search-response`（既定無効）は eval_search と同じ意味。既存の予算は変更していない。
bot の旧予算は candidates=16 / depth=10 なので、advisor と異なる。
対人戦向け予算とオフライン計測は [official_bot.md](official_bot.md) を参照。

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
`--search-response` を指定すると、次の公開の証拠が残る通常応答も探索する。

- 連続する受信で攻撃前の main 境界と、公開の launch カード・対象がある。
- 同じ受信内で自分の行動解決→GF進行→相手の launch が続く場合は、完全に解釈できた
  公開イベントの ledger の最後の攻撃前状態を使う。終点の MP から逆算しない。
- 証拠を保持したまま、公開の防御カード＋反射だけが続く場合は response journal を再生する。

C++ は公開 used 品と既に出されたカードだけを相手の anchor に置き、通常の
`Match::step` で launch/journal を再生する。確率攻撃では pending の存在が命中を、
202 では公開 amount が符号を示すので、その条件に合う新しい合成乱数列を使う。
サーバーの乱数や未使用の真在庫は受け取らない。goodfield 本体は変更していない。

根の全 `OBS_FIELDS`・mask・prefix・substep が `OfficialDecision` と完全一致する場合だけ
探索し、各 substep の選択前にも再照合する。根の構築で強制コマンドは実行しない。
探索状態は `main_public_worlds` / `response_public_worlds`。UI 除外後も、同じ手札スロット・kind・
敵既定カードの自己対象除外を根に適用して再探索し、`ui_mask_search` / `ui_retry_search` を返す。
除外は最初の決定の全 substep だけに適用し、後続 rollout は通常の合法手を使う。
世界生成では公開面・used・偽装有無・スロットの token を検証し、各 substep の制限後 mask を
公開カーソルと照合する。不一致なら `restricted_root_slot_mismatch` /
`restricted_root_world_mismatch` を理由に方策へ戻す。非対応時は従来の
`ui_mask_policy_fallback` 等と `search_fallback_reason` を残す。
Dream 最終手段も残り予算内で探索する（`dream_last_resort_search`）。1決定の探索は
初回＋再探索の最大2回で、それ以降は `search_limit` による方策 fallback とする。
`search_seconds` は初回と再探索の合計、`initial_search_status` は初回の状態。

再探索1回を要する固定 fixture の計測（CPU 1 thread、M `runs/play/m-2166.pt`、
AR `runs/play/belief-ar-m.pt`、worlds=16、depth=3、candidates=8、candidate-samples=64）：
ウォームアップ1回を除く5回の中央値 **0.419秒**、観測最大 **0.421秒**。
探索部分の累積最大は0.411秒（macOS arm64、PyTorch 2.15.0）。一般局面の上限を保証する値ではない。
再現: `.venv/bin/python scripts/bench_official_search.py --position ui-research --worlds 16 --depth 3 --candidates 8 --repeats 5 --out /tmp/ui-search.json`。

途中参加・欠番・不完全な前段イベントには anchor の証拠がなく、
`response_public_journal_unavailable_policy_fallback`。
追跡できない反射/跳ね返しは `response_redirect_journal_unavailable_policy_fallback`、
counter/death/guardian は `response_special_launch_policy_fallback`。
購入、Danger、霧/混乱、再生不一致、観測/prefix 不一致も方策へ戻す。
202 の符号が仮定値で警告される場合も、その警告を保持し、一致しない根は使わない。
強制手には `forced_no_inference`、応答探索の未指定には
`response_search_disabled_policy_fallback` を返す。

実 fixture は v73/update2、v68/update2・4、v78/update2 で根が一致した。
v59/update2・v89/update2 は選択が全て強制なので探索不要。
v73/update3 の counter/反射は通常 launch の証拠とは別の特殊 frame なので探索しない。
v64 の初回応答は攻撃前の受信がなく、方策を使う。
通常武器・確率攻撃・202・反射2回の合成局面も検査する。

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
  （40.5%）、終局4。応答待ちの選択は読み取り専用の `PublicSelection` を使う。
  上記の公開証拠がある場合だけ、別途実行用の根を通常 step から構築する。
  購入待ちは合成データでも検証。
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
- goodfield に main/response/purchase の合法 prefix 一致テストがある。
- uniform/AR について wire の相手の未使用モデル・数・fake・ID・名前を変えても、
  belief 入力（方策と同じ OBS_FIELDS のみ）、AR サンプル、候補値・確率・選択が不変。
  記録済み main/応答と合成応答を検査し、観測/prefix 不一致時の探索禁止も検査する。

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
