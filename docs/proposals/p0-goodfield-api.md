# goodfield に追加する P0 API の案（v0.1）

DESIGN.md §3.6 の「P0 で必要な API」を具体化したもの。目的は、gfrl_core（RL 側の C++）が JSON を経由せずに次の3つを得られるようにすること。

1. viewer ごとの観測（型付き）
2. 公開イベントの履歴を作る材料
3. 逐次選択（種別 → カード → 対象）の各段階の合法マスク

どれも C++ API として追加する。C ABI と Python は必要になってから足す。

---

## 1. 調べて分かったこと

### 合法性の構造（`src/official_match.cpp` の `prepare`、`src/official_rules.cpp` の `can_append_main` / `can_append_response`）

| 種別 | 合法な形 |
| --- | --- |
| play | 先頭は `can_start_main` を満たすカード。2枚目以降は「先頭が命中率0の武器なら +ATK カード」か「直前が奇跡なら FREE」だけ。+ATK は何枚でも続けられる（4枚上限は `legal_commands` の制限） |
| defend | 反射系は単品か、先頭から続く属性消去の後。普通の防具は攻撃に対して属性が合うもの。奇跡の直後に FREE。閃光中は1枚だけ |
| discard | 捨てられるカード（武器・復活・危険誘引以外）の空でない任意の集合 |
| pray | カードなし。見えている武器が手札にないときだけ |
| sell | [売る(4), 商品1枚] ＋ 生存者の対象 |
| buy | [買う(5)] ＋ 生存者の対象 |
| exchange | [両替カード] ＋ 配分（HP+MP+CP の合計を保存、各 0〜99） |
| accept / decline | purchase の局面だけ |

- **MP は最後にだけ検査される**。あとから入れる FREE が直前の奇跡の費用を消すので、途中の列で MP が足りなくても完成形では足りることがある。
- **対象**は列の中身で決まる。命中率つきは対象なし、全体攻撃・危険は任意、それ以外は生存者を明示（自分も可）。

### 順序が効果を変える（`attack_power` / `attack_element` / `magic_cost`）

- ×2 はその時点の攻撃力を倍にする。[武器10, +5, ×2] = 30、[武器10, ×2, +5] = 25。
- 属性の付け替え・消去は後に置いたものが勝つ。
- FREE は直前の奇跡の費用だけを消す。
- 応答の属性消去は先頭に続けて置く必要がある。

このため、カードを単純に集合として扱うと、実際にある選択肢が消える。

### 観測と公開イベント

- `observe()` の公開情報は JSON を組み立てる中でだけ作られている。pending の公開部分（霧の viewer には `forced_hit` を隠す処理など）や、`last_reveals` の owner_only の除外は、C++ の型としては取り出せない。
- 公式クライアントは update ごとに events を受け取る（記録済み wire の集計: gift 271件、advanceGF 83件、useAttackItems 75件、setTargetPlayer 58件、dealDamage 43件、useDefenseItems 22件 など）。
  - 他人への gift は届かない（手札の補充は本人にしか見えない）。
  - 使用イベントには実 modelId と fakeModelId の両方が入る（使った時点で偽装が明かされる。エンジンの `reveal_item` と同じ）。
  - 捨てる・祈るのイベントは記録にない。

---

## 2. API 1: 型付きの観測 `Match::view(viewer)`

```cpp
struct PendingView {
    std::optional<PlayerId> origin, source;   // 環境起因なら空
    PlayerId target = 0;
    std::vector<CardId> models;
    Category category = Category::weapons;
    Attribute element;
    int reflections = 0, bounces = 0, hit_rate = 0;
    bool counter = false, environmental = false, death = false;
    Guardian guardian = Guardian::none;
    std::optional<bool> forced_hit;           // 霧で隠れる viewer には空
    std::int64_t power = 0;
    Ability ability = Ability::none;
    int amount = 0;
};
struct PurchaseOfferView { PlayerId origin = 0, buyer = 0, seller = 0; CardId model = 0; int price = 0; };
struct MatchView {
    MatchPhase phase = MatchPhase::finished;
    PlayerId owner = 0;
    std::optional<PlayerId> actor;
    std::uint64_t decisions = 0;
    bool terminated = false, truncated = false, apocalypse_active = false;
    SnapshotView board;                        // 既存の project_view(state, viewer)
    std::optional<PendingView> pending;
    std::optional<PurchaseOfferView> purchase;
    // 現在の main action 中の公開イベント。observe() の last_* と同じ内容を viewer で除外済みにしたもの
    std::vector<RevealedItem> last_reveals;
    std::vector<PlayerId> last_redraws;
    std::vector<GuardianChange> last_guardian_changes;
    std::vector<GuardianAction> last_auto_guardians;
    std::optional<DangerEvent> last_danger;
    std::optional<RemovalEvent> last_removal;
    std::optional<SaleEvent> last_sale;
    std::optional<PurchaseEvent> last_purchase;
    // last_grant / last_devil / last_fate / last_phenomenon / last_guardian / last_guardian_payload なども同様
};
[[nodiscard]] MatchView Match::view(std::optional<PlayerId> viewer) const;
```

- `observe(viewer)` は、この view を JSON にしたもの（＋ルール文字列）と同じ内容にする。ランダムな局面・全 viewer（観戦者を含む）で一致することをテストする。
- 余裕があれば `observe()` を `view()` の上に作り直し、公開範囲の判定を1か所にまとめる（既存テストが安全網になる）。最初は並べて追加するだけでもよい。

---

## 3. API 2: 直前の決定の公開記録 `Match::last_decision(viewer)`

```cpp
enum class DecisionSource : std::uint8_t {
    player, confusion, fate, devil, grant, artifact_grant, guardian, guardian_payload,
    phenomenon, item_shuffle, redraw_response, purchase_response, summon_result, automatic_guardian_response
};
struct PublicDecision {
    std::uint64_t decision = 0;               // 受理後の通算決定番号
    std::int64_t gf = 0;                      // 受理した時点の GF
    MatchPhase phase = MatchPhase::main;      // 受理した時点の phase
    DecisionSource source = DecisionSource::player;
    std::optional<PlayerId> actor;            // 環境起因なら空
    std::optional<CommandKind> kind;          // player / confusion のとき
    std::vector<CardId> faces;                // 場に出たカードの見かけ ID（選択順）
    bool faces_hidden = false;                // viewer にはカードが見えない
    std::optional<PlayerId> target;           // 公開表示と同じ対象（霧で選び直された場合は選び直した後）
    std::optional<ResourceAllocation> allocation; // exchange
    std::optional<CardId> outcome_model;      // fate の現象、devil、phenomenon などの公開される結果
};
[[nodiscard]] std::optional<PublicDecision> Match::last_decision(std::optional<PlayerId> viewer) const;
```

- **結果（HP の増減・災い・死亡など）はこの記録に入れない**。gfrl_core が「決定前の `view(v)`」と「決定後の `view(v)`」の差分として作る。どちらも viewer で除外済みなので、差分からも情報は漏れない。
  - 他人の手札の補充は view に出ないので、差分にも出ない。自分の補充は自分の手札の差分として出る。
  - 霧で隠れた欄は前後とも空なので、差分は「不明」になる。
- `MatchSnapshot` に「直前の決定の記録」を1つ追加する（snapshot の版を 72 に上げる）。clone すれば記録もついてくる。
- 見える範囲:

| 決定 | 見えるもの | 根拠 |
| --- | --- | --- |
| play / defend | faces、対象 | 公式 wire の useAttackItems / useDefenseItems |
| sell / buy / exchange | faces、対象、配分 | wire の buy / exchange、既存の last_sale / last_purchase |
| accept / decline | 種別だけ（結果は last_purchase） | 既存の purchase 表示 |
| pray | 祈ったこと。祈って引いたカードは見えない | ユーザー確認（2026-11-06）。公開表示 `ActiveAttackView` にも `prayer` がある |
| discard | faces（捨てたカードの種類）。ただし viewer が霧のときは、他人の捨札は見えない（`faces_hidden`） | ユーザー確認（2026-11-06）。霧の扱いは推測で未観測 |
| チャンス系（fate / devil / grant など） | 既存の last_* と同じ範囲 | 既存の view 契約 |

---

## 4. API 3: 逐次選択の合法マスク `Match::legal_extensions`

```cpp
struct ExchangeDomain { std::int64_t total = 0; };      // hp + mp + cp == total、各 0..99
struct SelectionNode {
    std::vector<ItemToken> next;                         // 追加すると合法な完成形に到達できる token
    bool can_finish = false;                             // いまの列で確定できる
    std::vector<std::optional<PlayerId>> targets;        // can_finish のとき選べる対象（nullopt = 対象なし）
    std::optional<ExchangeDomain> exchange;              // kind == exchange かつ can_finish のとき
};
struct SelectionOptions { bool canonical = false; };
[[nodiscard]] std::vector<CommandKind> Match::legal_kinds() const;  // 合法な Command が1つ以上ある種別
[[nodiscard]] SelectionNode Match::legal_extensions(CommandKind kind, std::span<const ItemToken> prefix,
                                                    SelectionOptions options = {}) const;
```

### 契約

- `canonical = false` のとき、任意の Command c について次が同値:
  - `legal(c)` が true
  - c.kind ∈ `legal_kinds()`、各カードがそれまでの prefix に対する `next` に含まれる、最後の prefix で `can_finish`、c.target ∈ `targets`、配分が domain 内
- 状態と乱数を変えない。混乱中の main や grant など専用入口の局面では、既存の `legal_commands` と同じく例外を投げる。
- MP は「この先 FREE で消せる費用」を見込んだ下限で判定し、確定時に完全に判定する。
- 1回あたりの計算は手札の枚数に比例する程度（目安は数 μs）。

### `canonical = true`（正準化。A2 で決定）

効果が同じになると確かめられる場合だけ、並べ方を1つに絞る。規則:

1. 見た目が同じ手札（face・used・disguised が同じ）は1つにまとめ、一番小さい token を代表にする。
2. 捨てるは集合として扱い、手札の並び順の昇順だけを許す。
3. 攻撃や防御の追加カードのうち、入れ替えても攻撃力・属性・MP 費用・能力が変わらない連続した区間（×2・属性付け替え・属性消去・FREE を含まない区間）は、カード ID の昇順だけを許す。

### テスト

1. `canonical = false` で、ランダム局面の `legal_commands(max_items = 4)` と、木をたどって得られる4枚以下の Command の集合が一致する。
2. 5枚以上: 木からランダムにたどった Command はすべて `legal()` が true。`sample_action` が出す Command はすべて木の上にある。
3. `canonical = true` の木から得られる「効果シグネチャ」（攻撃力・属性・MP 費用・モデルの多重集合・能力・対象）の集合が、`canonical = false` のものと一致する。

---

## 5. 相談事項

| ID | 論点 | 結論 | 状態 |
| --- | --- | --- | --- |
| A1 | 公開イベントの粒度 | 決定単位（API 2）＋前後の view の差分。公式 wire のイベント語彙（効果単位）には合わせない | 決定 |
| A2 | 順序の正準化 | 効果が同じと確かめられる場合だけ正準化する（§4 の3規則）。テスト3で効果が失われないことを確認する | 決定 |
| A3 | 捨てる・祈るの公開範囲 | 祈り: 祈ったことは見え、引いたカードは見えない。捨てる: 捨てたカードは見える。viewer が霧のときは見えない（推測） | 決定（霧は未観測） |

### A1 の比較

| | 決定単位＋差分（推奨） | 公式 wire 語彙で効果単位 |
| --- | --- | --- |
| goodfield の作業 | `view` と `last_decision` だけ | 全効果で公式と同じ順序・内容のイベントを出す。サーバーのイベント生成規則は未取得 |
| 情報量 | 決定ごとの見える変化はすべて入る。1つの決定の中の順序（反射→ダメージ→復活など）はまとめられる | 1つの決定の中の順序まで分かる |
| 本家との接続 | 受信 update の境界で同じ形に変換できる | そのまま使える |

---

## 6. gfrl_core 側でやること（参考）

- 各決定の前後で両プレイヤーの `view` を取り、`last_decision` と差分から履歴イベントを作ってリングバッファに積む。
- 履歴と観測をトークン化し、テンソルに書く。
- `legal_kinds` / `legal_extensions(canonical = true)` で各段階のマスクを作る。

## 7. 進め方

合意したら、`view` → `last_decision` → `legal_extensions` の順に1つずつ codex exec に渡す。それぞれテストを含め、goodfield の既存テスト（Release / ASan）が通ることを完了条件にする。
