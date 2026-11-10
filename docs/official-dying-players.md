# 受信 game.dyingPlayerIds の取り込み

ネットワークを使わず、保存済み資料と公式 client を調査した。
公式 client は `/Users/yuto/godfield-bot/data/snapshots/2026-11-03/official-client.js`、
SHA256 は `5ece06c224b6a7a5d877eb45d22ea18b82a03aaf1b7161cbbdf6a1e23ceab2cd`。
goodfield の `scripts/official_client_oracle.cjs` と `data/official-v2/manifest.json`
に固定された v2 と一致する。client 自体はリポジトリへコピーしない。

この client に `dyingPlayerIds` という文字列は無い。game decoder `tn`
（3754–3789 行付近）は players、GF、手番、先頭 attack、isOver、events、updateCount
を読むが、この key を読まない。したがって、client から配列の生成条件や死亡処理順を証明することはできない。
`official_client_oracle.cjs` の `loadPinnedClient` で DOM・fetch・通信のない VM に読み込み、
合成 game の `dyingPlayerIds` なし / `[2]` ありを `tn(nI(...))` で復号して比較した結果も同一だった。
auto46 の停止記録から確認できるのは game 直下にこの key が出現したことまで。
名前と出現状況から死亡処理中の player 集合を示す server metadata と解釈するが、
HP0、dead 確定、復活予定、昇天弓発火のいずれとも同一視しない。

`third_party/goodfield/docs/official-current-rules.md` の「太陽のお守り：HP0と脱落の境界」
および「昇天弓112」では、HP変化、revive、die、attackDyingly は別の受信処理。
client の `attackDyingly` handler（11705–11749 行付近）は指定された source、item、
atk、hitRate を表示・消費する。分岐の記述順は server の発行順を示さない。
`legacy-revival-rules.md` の旧244の走査順・旧117の発火条件は現行の証拠にはしない。
auto44 / roomA13 との差が手札や死亡処理順によるものかは、これらの資料だけでは確定しない。

取り込み契約は game 直下の配列に限定し、各要素は同じ game の roster に存在する厳密な整数 ID、
重複なしとする。空配列も許す。bool、float、文字列、null、別 roster の ID、入れ子は拒否する。
これは server の完全な schema を証明したものではなく、許可する最小の形である。

投影では非空配列を公開の `dying_players` として保持する。dead や HP は変更しない。
native の PendingView には death / hit_rate があり、現在の投影は death=False、
台帳の通常 hit_rate を使う。ID 配列だけでは死亡時攻撃の値や未解決 queue を復元できず、
同じ観測・mask になることを保証できない。自分の決定がある場合だけ
`dying_players_decision_unavailable`、理由 `native_death_state_not_reconstructible`
で停止する。自分の決定が無い場合は既存の decision-only 判定に従って待機・次更新を取り込む。
終局は finished。空配列は省略時と同じ投影・観測・mask とし、合成テストで比較する。
未知イベントなど別の未対応理由の判定はそのまま維持する。
