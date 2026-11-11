# 保存済み client に基づくイベント取り込み（2 回目）

ネットワークは使用しない。根拠は goodfield の
`docs/official-current-rules.md`、`scripts/official_phenomenon_oracle.cjs`、
`scripts/official_targeted_phenomenon_oracle.cjs` と保存済みの固定 client。
client SHA256 は `data/official-v2/manifest.json` と一致する
`5ece06c224b6a7a5d877eb45d22ea18b82a03aaf1b7161cbbdf6a1e23ceab2cd`。
本文・個人情報・画像はこの資料にも fixture にも含めない。

## 検査と置換の原因

- `sell`: 従来の追加検査は `overflowItem` キーがあると値を完全な
  `{id, modelId, fakeModelId, used}` 辞書に限定し、`null` を拒否していた。
  `A.to`（3541 行付近）は null をそのまま読み、`ks.bE/eH`（11863 行付近）は
  null なら何もしない。null は「溢れなし」。商品には既存の compact item 形を
  維持し、存在する各フィールドの型と範囲を検査する。非 null overflow は完全な
  item のみ。最近の指定 run には raw 受信本文が残っておらず、警告の keys だけで
  この 5 件が null だったとは断定できない。完全な item 2 個なら従来から受理される。
- `removeItems`: 既存の 2 箇所の検査が `playerId` を必須にしていた。
  固定 client の受信分岐（11520 行付近）は現在の対象 pane を選び、
  event.items だけを `dr/c2/kq` に渡す。`playerId` なしを厳密に追加し、
  既存の playerId 付き形も保持する。items は完全な item 辞書の配列。
- `counterAttack` / `selfCurse`: `index` が key allowlist に無かった。
  `ks.k`（11614–11641 行付近）は応答側の `a6()` の選択品を index で光らせる。
  attacks 配列の添字ではない。整数 0..17、同じ受信内に `useDefenseItems` があれば
  その枚数未満。選択記録が前の受信にあれば現在手札や attacks から枚数を推定しない。
  既存の payload なし marker（client の index default 0）も維持する。
- 残る 4 action は文字列 allowlist に無かった。各 action の下記の keys だけを許可する。
  playerId は同じ game の roster 整数、damage は bool/float を除く整数、item は
  id>=0 / modelId 1..296 / fakeModelId 0..296 / used bool の完全な辞書。
  追加キーは redaction 前に拒否する。

## 公開記録と History

| event | 意味と公開記録 | History |
|---|---|---|
| sell | 配送。null overflow は品の排出なし。相手の商品・未使用 overflow の真値は転送しない | 既存の Sale/価格/資源差分。未観測 overflow・偽装は incomplete |
| removeItems | 現在対象の指定品を除去。client はその品を表示し、偽装なら flip する | 受信済み launch/対象が分かる場合は既存の Removal。同じ native テストを playerId なしでも照合。偽装・起点不明は incomplete |
| counterAttack | 指定した応答品の点滅 | 独立決定や反撃は生成しない。後続 nextAttack を使う |
| selfCurse | 指定した応答品の点滅 | marker 自体は病気/HPを変えない。別の addCurse 等を使う。既存の病気・視界 journal の不足は incomplete のまま |
| dealDarkDamage | 現在対象の HP を 0 にする（11287 行付近）。damage 値を減算しない | 既存の決定の HP 差分。死亡/守護神離脱/復活は別 event。決定起点不明は incomplete |
| upgradeHeaven | 天国進行の演出（11443 行付近、cl）。この handler は HP や病気を変更しない | 別の数値/死亡 event を既存決定に反映。marker 自体の独立 token はない。決定起点不明は incomplete |
| revive | 指定 player の消費した復活品を公開表示。偽装は flip（11642 行付近） | 通常の 208 消費は別 boostHP の資源差分へ。HP回復・dead取消を推定しない。偽装 Reveal journal・決定起点が不足すれば incomplete |
| collectCPOfEverybody | 生存者全員の CP を集め、全員を 0 にして指定 player に加算、99 cap（11353 行付近） | 公開受取人だけを残す。native の外部 phenomenon 決定の source/anchor と journal を marker だけから復元できず、常に incomplete。非公開 CP 合計を出力しない |

revive では公開された消費品だけを投影し、同じ player の他の未使用品、instance ID、
受信 inventory の長さを転送しない。overflow は既存の保守的な公開範囲を維持する。
相手の未使用 inventory や品の ID を変えても公開投影が変化しないことを合成テストで確認する。

`tests/python/test_official_received_events.py` は闇→208消費→HP回復と天国進行→死亡を
実際の native Match で生成し、両視点の観測/History 全体を照合する。
parser の合成 Firestore frame は既定 strict と lenient の両方で unknownEvent への置換が
ないこと、型・キー・roster・応答 index の不正を拒否することを確認する。
