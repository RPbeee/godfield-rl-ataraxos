# 受信 attack の hitRate（オフライン調査）

## 値と client の扱い

- [現行仕様資料](../third_party/goodfield/docs/official-current-rules.md) の確率武器節、死亡時攻撃節、v48 active attack 節を参照。命中率は百分率で、台帳には 25/50/75 がある。112 昇天弓は通常 Light ATK1/25%、説明上の死亡時攻撃は ATK30/75%。
- [active attack oracle](../third_party/goodfield/scripts/official_active_attack_oracle.cjs) と [保存 report](../third_party/goodfield/data/official-v2/active-attack-client-v48.json) は受信 DTO decoder と表示経路を検査している。summary は正の `hitRate` を優先し、0/欠損では基底台帳値へ戻る。整数 73/81 などの人工 override も表示検査に使われている。client の不正型 coercion の例は、厳密な受信境界で許可する形の根拠にはしない。
- [守護神 payload oracle](../third_party/goodfield/scripts/official_guardian_payload_oracle.cjs) と [保存 report](../third_party/goodfield/data/official-v2/guardian-payload-client-v33.json) の decoder も任意 DTO の atk/hitRate を読む。Earth/Moon でキーが来る可能性はあるが、表示できることは任意 override を実行器が生成する証拠ではない。
- `attackDyingly` の受信表示は server 指定の source/atk/hitRate/item を使う。通常の確率武器や守護神を含む active DTO でも動的値を表示できる。ただし、保存 client 資料だけでは、server がどの攻撃でこのキーを必ず付けるか、または省略するかは確定できない。

受信境界では `game.attacks[]` に限り **exact int、0〜100（両端含む）** を許可する。bool、float、文字列、null は不可。これは百分率としての厳密な受理範囲であり、server が全整数値を発行するという主張ではない。event や他の場所への allowlist 拡張は行わない。イベントの寛容取り込みを有効にしても、不正な authoritative attacks は拒否する。

## native 投影の判断

[native adapter](../cpp/gfrl_core/official_bindings.hpp) は `PendingView.hit_rate` を基底台帳から復元する。[encoder](../cpp/gfrl_core/encoder.cpp) はそれを 100 で割った公開観測を作る。したがって、正の受信値が台帳と一致する場合、および 0/欠損の場合は従来の観測・mask を維持できる。

異なる正の値は、この adapter のままでは native と同じ観測を作れない。台帳値で判断を続けず、`dynamic_hit_rate_unavailable` と受信・台帳の公開命中率を記録する。自分の決定時は停止し、それ以外では待機する。active head だけを判断し、後続 queue の動的値を現在の決定へ適用しない。

[native 死亡処理](../third_party/goodfield/src/official_match.cpp) の昇天弓は power30/rate75 と `is_death=true` を持つ。受信 atk30 または rate75 だけでは死亡処理の由来やスケジュールを確定できないため、112 の該当形には `death_attack_decision_unavailable` も記録する。通常の atk1/rate25 は従来どおり扱う。既存の `dyingPlayerIds` による停止理由は維持する。

## auto54 の再現範囲

記録の停止 state と行動ログから GF35、主動作前の HP7/MP0/CP2、相手 HP11/MP15/CP0、112/106 を含む公開手札、model44 の使用選択を確認した。受信本文は存在しないため、次の DTO の model/source/target/atk/hitRate や死亡時攻撃だったかは確定できない。

[回帰テスト](../tests/python/test_official_attack_hit_rate.py) は識別情報を含まない合成 state と、106 の 75%、112 の通常25%・死亡時候補75%を両 target で検査する。実受信の再生や停止原因の攻撃種別の断定ではない。raw 受信、認証情報、名前、画面画像は保存しない。学習・収集・評価の実行や規則変更も行わない。
