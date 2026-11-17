# 受信 attack の hitRate（オフライン調査）

## 値と client の扱い

- [現行仕様資料](../third_party/goodfield/docs/official-current-rules.md) の確率武器節、死亡時攻撃節、v48 active attack 節を参照。命中率は百分率で、台帳には 25/50/75 がある。112 昇天弓は通常 Light ATK1/25%、説明上の死亡時攻撃は ATK30/75%。
- [active attack oracle](../third_party/goodfield/scripts/official_active_attack_oracle.cjs) と [保存 report](../third_party/goodfield/data/official-v2/active-attack-client-v48.json) は受信 DTO decoder と表示経路を検査している。summary は正の `hitRate` を優先し、0/欠損では基底台帳値へ戻る。整数 73/81 などの人工 override も表示検査に使われている。client の不正型 coercion の例は、厳密な受信境界で許可する形の根拠にはしない。
- [守護神 payload oracle](../third_party/goodfield/scripts/official_guardian_payload_oracle.cjs) と [保存 report](../third_party/goodfield/data/official-v2/guardian-payload-client-v33.json) の decoder も任意 DTO の atk/hitRate を読む。Earth/Moon でキーが来る可能性はあるが、表示できることは任意 override を実行器が生成する証拠ではない。
- `attackDyingly` の受信表示は server 指定の source/atk/hitRate/item を使う。通常の確率武器や守護神を含む active DTO でも動的値を表示できる。ただし、保存 client 資料だけでは、server がどの攻撃でこのキーを必ず付けるか、または省略するかは確定できない。

受信境界では `game.attacks[]` に限り **exact int、0〜100（両端含む）** を許可する。bool、float、文字列、null は不可。これは百分率としての厳密な受理範囲であり、server が全整数値を発行するという主張ではない。event や他の場所への allowlist 拡張は行わない。イベントの寛容取り込みを有効にしても、不正な authoritative attacks は拒否する。

## native 投影の判断

[native adapter](../cpp/gfrl_core/official_bindings.hpp) は `PendingView.hit_rate` に正の受信 override を使い、省略時は基底台帳から復元する。[encoder](../cpp/gfrl_core/encoder.cpp) はそれを 100 で割った公開観測を作る。したがって、正の受信値が台帳と一致する場合、および 0/欠損の場合は従来の観測・mask を維持できる。

異なる正の値は public pending の `hit_rate` として native の観測へ反映する。
実行時の発動来歴は確定しないため `dynamic_hit_rate_unavailable` の診断を残し、
条件付きで `_policy_only` へ移して現在の防御を続行する。0/欠損/台帳と同じ値は従来どおり。
active head だけを判断し、後続 queue の動的値を現在の決定へ適用しない。

112 の atk30 または rate75 は `death_attack_decision_unavailable` の候補として扱う。
死亡 queue や `death=True` を推測せず、現在の head に正の atk が明示され、自分が
live・HP正・dying でなく、他の未対応理由が無い場合は方策で応答する。
rate75のみで atk が欠損した場合は停止する。直前の致死攻撃の来歴も流用しない。
探索は Python・C++ の両方で遮断し、発動の強制命中は不明として観測する。
詳細と停止条件は [途中停止対策・第2弾](official-bot-stop-recovery.md) を参照。

## auto54 の再現範囲

記録の停止 state と行動ログから GF35、主動作前の HP7/MP0/CP2、相手 HP11/MP15/CP0、112/106 を含む公開手札、model44 の使用選択を確認した。受信本文は存在しないため、次の DTO の model/source/target/atk/hitRate や死亡時攻撃だったかは確定できない。

[回帰テスト](../tests/python/test_official_attack_hit_rate.py) は識別情報を含まない合成 state と、106 の 75%、112 の通常25%・死亡時候補75%を両 target で検査する。実受信の再生や停止原因の攻撃種別の断定ではない。raw 受信、認証情報、名前、画面画像は保存しない。学習・収集・評価の実行や規則変更も行わない。
