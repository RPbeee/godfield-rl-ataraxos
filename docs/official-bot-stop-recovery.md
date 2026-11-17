# 練習 bot の途中停止対策（オフライン）

保存済みの停止行・画面 state と、固定 v2 client および goodfield の公開選択実装を
照合した。公式サイト・ネットワークには接続していない。受信本文は保存記録に無いため、
実受信の再生や実戦での改善率の測定ではない。回帰テストは公開情報を合成して生成し、
実際のログ・画面・人物名・認証情報を含めない。

## 選択不一致

記録の12件は防御。クリックしたカードは手札に残り、自己選択欄は空だった。
多くは手札・攻撃カード・資源が不変で、中央の文字だけが変わっていた。
確率攻撃の表示や「許す」が完成する前でも、旧 ready の画像・資源比較は通る。
client はイベント演出を処理してから fF で command を作り、手札の選択受付を有効にする。
このタイミング差は記録に整合するが、過去のクリック時点の受付状態を直接観測した証拠ではない。

probe に `commandPanes` を追加した。ut が作る透明な command ボタンを、可視 DOM の
矩形、z-index、cursor、pointer-events、opacity、ヒットテストで特定する。
アプリ状態・ハンドラ・非公開情報は読まない。選択・確認は自己 pane のボタンがある場合だけ行う。
全防御で新しい連続2状態の安定も確認し、その後の座標で plan を作り直す。
過去のオフライン snapshot はこの能力を持たず、従来の検証を維持する。

操作不能や表示不安定が settle-timeout を超えた場合は停止する。
余分な選択、対象・矢印・資源・手札の変化を無視して再クリックしない。
受信が進んだら古い操作を中断して新しい決定から助言し直す。
従来の「同一状態で無反応だったクリックを1回だけ再試行する」制限も維持する。

## dying / redirect / multiple frames

`dying_players_decision_unavailable` は死亡処理の実行状態を再構成できないという理由だった。
ただし、現在の防御の合法性と、その後の実行順の復元は必要な情報が異なる。
response_context / PublicSelection が防御の合法性に使うのは現在の category / element /
power と、自分の可視手札・MP・Flash。相手の未使用在庫、死亡 queue、強制命中フラグは使わない。

自分が dying 集合に含まれず、公開 HP が正、dead=False、他の未対応理由が無い防御は
方策で続行する。ID 配列から死亡時112や復活の予定を推測しない。
自分が dying / HP0、死亡処理中の主動作、威力不足などは停止する。
動的 rate・死亡時112 の現在の防御は、第2弾の条件付き方策対応へ拡張した（後述）。

`redirect_chance_activation_unavailable` は反射/跳ね返し後の発動来歴が不明なためだった。
forced_hit=None として不明性を観測へ残し、公開カーソルで防御を選ぶ。
威力や対象まで不明な場合は救済しない。

複数 attack は既に公式 client と同じ先頭だけで応答する実装になっていた。
今回の複数 frame 停止には dying の理由も併存していた。後続 frame に由来する
イベントで先頭の provenance を上書きせず、方策だけで現在の先頭へ応答する。
後続 frame の攻撃値や対象を現在の防御判断へ流用しない。

これらは `_policy_only` と固定コードの警告・detail を残す。探索前に方策へ戻し、
C++ の探索根生成にも拒否条件を設ける。次の受信では条件を判定し直す。
受信の破損・未対応の schema・対象不明・conflicting update は引き続き停止する。
既知の公開キー confusionCount は第2弾でパーサーへ追加した。

## 交換の途中範囲外

旧実装は MP/CP の減額を先に全部行い、HP を移動のバッファとしていた。
合計が99を超える場合、合法な最終配分でも途中の HP が99を超えることがある。
HP の余地に合わせて増減を交互に行い、10または1の操作を選ぶようにした。
各資源は常に0〜99、合計は保存し、各クリック後と最後の preview を従来どおり検証する。
不正な最終配分・合計不一致・UI による値の相違は停止する。配分クリックは再送しない。

## 使用済み奇跡の位置

旧実装は使用済みが全体で1枚のときだけ右下の使用済みコピーを区別していた。
client の fH.hr は使用済みを model 順に並べ、fH.fD は18枠の右下へ詰める。
複数の使用済みについて、その suffix 全体の座標と face を検証してから
使用済み/未使用の同一 face を区別するようにした。
選択前の preflight を再読込待機に含め、移動中の位置ではクリックしない。

suffix が一致しない、選択した face の使用済みが複数あり個体を識別できない、
同じ face に偽装品があるなどの場合は、推測せず停止する。

## 受理タイムアウト

記録では確認後に選択欄が空になり、GF は変わらず、受信更新は確認できなかった。
client は submit-command の待機前に command ボタンと選択欄を消す。
送信失敗・受信遅延・通信停止のどれだったかは記録だけでは断定できない。

新 probe で「確認直前は自己 command ボタンあり → 今は無し、選択欄も空、
同じ GF/自己 pane」という遷移を確認できた場合、max(timeout,30秒) の
受信だけの猶予を1回追加する。`acceptance_wait_extended` を記録する。
command を再構成・再選択しない。受理には authoritative update の前進を要求する。
ボタンと選択が同一のままなら従来の確認再試行1回を使う。
猶予後も更新が無ければ停止する。第2弾では、確認前の command ボタンを
観測済みで、その後に画面全体が空白になった場合にも同じ受信だけの猶予を適用する。

## 検証範囲

`tests/python/test_official_stop_recovery.py` は入力受付と演出のずれ、受信進行時の中断、
複数使用済みの完全な suffix、交換の境界例と1000件の配分、方策への切替と探索遮断、
非公開在庫を変えても投影・観測・mask・行動が変わらないこと、送信後の有限な受信待機を検証する。
既存の安全停止・UI・受信・探索テストも実行する。1局中に停止する割合の変化は
実戦での別の検証が必要。

第1弾の検証（macOS arm64）: 指定の Release ビルド成功、`ctest --test-dir build` は
18/18成功。指定の MPS equivalence 1件を除外した pytest は1460 passed / 122 skipped /
1 deselected。新規合成テスト46件を含む。既存 transfer.py で読み取り専用 NumPy 配列を
PyTorch に渡す警告が1件出た。Node の2スクリプトの構文検査と `git diff --check` も成功。


## 第2弾: 40局中8停止の調査と対処

ネットワーク・公式サイト・実対局は使用せず、指定の停止記録8件と固定 client、
goodfield の公開選択コードを読んだ。受信本文は記録に無いため、実データの再生や
停止率の改善を実測したものではない。追加テストも公開状態だけの合成データである。
`third_party/goodfield`、他の worktree、学習器・oracle の接続は変更していない。

### 動的命中率 / 死亡時112 / dying の同時発生（5件）

原因は、現在の防御を選べるかという条件と、死亡 queue を同じ順序で実行できるかという
条件を一緒に判定していたこと。記録では112・75%・他プレイヤーの dying が併存していた。
通常の112は25%で、旧 adapter は PendingView の命中率を常に台帳から作っていた。

正の受信 hitRate が台帳と異なる場合、public pending の `hit_rate` を通して native の
観測にも反映する。0/欠損/台帳と同じ値は従来の観測・mask と同一。
発動来歴は `forced_hit=None` とし、`death=True` や死亡順序を推測しない。
`dynamic_hit_rate_unavailable` / `death_attack_decision_unavailable` は診断コードとして残し、
安全に応答できるときは unsupported から `_policy_only` へ移す。

継続条件は response の現在の head の source / target / model が確定し、自分が
live・HP正・dying でなく、他の未対応理由が無いこと。死亡時112の候補には、さらに
**現在の authoritative head に正の atk が明示されていること**を要求する。
75%だけからATK30を補ったり、欠損時に通常ATK1で応答したりしない。
直前の自分の致死攻撃や別 frame の source / power / redirect の履歴を死亡時攻撃へ
流用しない。死亡候補の head では launch provenance を破棄する。

`PublicSelection` の response は category / element / damaging と自分の手札・MP・Flash
で合法性を作る。hit_rate / death / forced_hit / 相手の未使用在庫 / 未解決の死亡 queue は
合法性の入力ではない。方策でこの cursor の合法手を選び、従来のカード・対象・GF・資源・
command ボタンと安定した画面の照合を全て通してから操作する。
Python の助言は探索を呼ばず、C++ の探索根生成も `_policy_only` を拒否する。
marker を外した public state でも、台帳と異なる命中率の探索根を拒否する。

自分が dying / HP0 / dead、死亡処理中の主動作、対象不明、死亡候補の atk 不足、
未対応の別効果、来歴を確認できない reversed などは引き続き停止する。
原記録に受信本文が無いため、5件すべてに atk が明示されていたとは断定しない。
次更新で通常状態へ戻れば `_policy_only` を消し、通常の探索条件を再判定する。

### confusionCount の未知キー（1件）

固定 client の player decoder は `confusionCount` を整数として読み、状態パネルに
混乱の枠を描く。投影側は元々 `confused` を持っていたが、vendored parser の
allowlist にキーが無く、その前で拒否されていた。

vendored ファイルを変更せず、`parser_module()` のローカルな検証拡張で
`game.players[]` に限り exact int・0〜JavaScript safe integer上限を受理する。
bool / float / 負数 / null / 配列などは拒否し、他の位置の同名キーや他の未知キーも許可しない。
イベント寛容取り込みでも authoritative player の検証を省略しない。

判断へ渡すのは可視の `confused` 真偽だけ。残り回数は渡さず、相手が霧なら隠し、
dead なら false。混乱のパネル枠は pointer-events:none で、カードの選択位置を変えない。
相手の混乱や自分の response は既存の厳密な UI 照合を保って続行する。
可視の混乱があれば、残り回数を仮定して探索せず方策へ戻す。

自分の混乱した main は通常の合法手選択を構築せず、次の受信更新を待つ。
自動解決の内容を生成・送信しない。次更新が来なければ既定の有限な受信待機後に
acceptance_timeout で止まる。別の不正 schema / 未対応理由は待機に置き換えない。

### visible_state_timeout（1件）

保存画面は開始前の「修行」の枠とヘッダー・フッターだけで、設定・手札・GFが無かった。
旧 enter_training は設定の表示を固定20秒で待っていた。
通信や描画がなぜ進まなかったかは記録だけでは断定できない。

設定を待つ基本時間に settle_timeout を使い、期限時にこの空の修行枠、または完全な空白画面
を確認できた場合は1回だけ `max(timeout,30秒)` を追加する。
修行や開始ボタンを再クリックしない。設定の検証に通って初めて次へ進む。
猶予後も表示が無ければ同じ visible_state_timeout で停止し、detail に wait_extended を残す。
未知画面、設定の人数・終末条件不一致、進行しない接続障害を推測で突破しない。

### acceptance_timeout（1件）

交換を選択・配分・確認した後、保存画面は全面空白だった。
旧 submission_pending は同じGFと自己 pane が残ることを要求したため、既存の追加待機に
入れなかった。原因が reload / 描画遅延 / 通信断のどれかは確定できない。

確認直前の自己 command ボタンを観測済みで、body / texts / images / inputs が空、
commandPanes=[] の場合にも、既存と同じ1回だけの受信猶予を適用する。
`acceptance_wait_extended` と `acceptance_blank_wait` を成功行・停止行に残す。
空白からの復帰だけでは受理扱いにせず、authoritative update の前進を要求する。
受信後の次の操作も ready / prepare から検証し直す。

猶予後も更新が無ければ停止する。確認前のボタンを未観測、部分的にカードや文字が残る
状態には空白猶予を適用しない。リロード、配分・カードの再選択、送信の再構築は行わない。
従来の完全一致した確認状態だけの再試行1回や、空防御だけの「許す」の制約は維持する。

### 第2弾の検証

追加の合成テストで、ATK30 / 75%の観測、response の全選択 prefix の合法性と
rate / death / forced_hit の独立性、探索の遮断、死亡候補の来歴分離、危険な境界の停止、
相手の非公開在庫を変えても投影・観測・mask・行動が不変であることを検証する。
confusionCount の型・位置・霧・死亡・main待機・response継続と、遅延した修行枠・
送信後の空白からの復帰・無期限待機の防止・再送しないことも検証する。
既存テストのうち拡張した対応範囲の期待値を更新し、他の停止条件は維持した。

最終検証（macOS arm64）: 指定の Release configure / build 成功。
`ctest --test-dir build` は18/18成功。指定の MPS equivalence 1件を除外した pytest は
1535 passed / 122 skipped / 1 deselected。新規の合成テスト55件を含む。
既存 transfer.py の読み取り専用 NumPy 配列を PyTorch に渡す警告が1件出た。
`git diff --check` 成功、third_party/goodfield の差分なし。コミットは行っていない。
実戦での停止率改善は未測定で、Claude による対局確認が必要。
