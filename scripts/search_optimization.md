# 探索評価の最適化計測（2026-11-09）

変更前は作業開始時の `8459231` の search.py / actor.py。macOS 27.0.1 arm64、PyTorch 2.15.0、CPU 1 thread。MPS は `torch.backends.mps.is_available() == False`、CUDA は sandbox で使用不可。GPU の速度・数値一致は未測定。

既存 S checkpoint `runs/m4-s-ppo-v1/checkpoint.pt`（d128 / encoder depth4）と B1 (`runs/belief-smoke/fit.pt`) を使用。モデルロード・局面収集を除外し、各条件を warmup 後、同一根・同一 seed で5回計測。実行順をローテーションした中央値。CPU を使う他のテストとの同時実行は避けた。

Small は2根、worlds8 / depth1 / candidate-samples16。C 相当は4根、worlds64 / depth3 / candidates16 / candidate-samples64 / beta0.1 / leaf-view actor / max-inference-batch8192。いずれも policy S。C 相当は16環境を方策で20 stage進めて取れた最初の4つの対応可能な根。

**この S 局面の候補数は平均1.5個（うち単一候補50%）で、本番 M の平均約5個を代表しない。CPU の結果を H100 MIG に外挿しない。数倍という目標は、複数候補を含むこの計測では未達。**

## 各最適化（秒 / 探索呼び出し）

`host` はキャッシュのコピー処理だけの変更、`cache` はそれに device/auto 配置を追加。`forced` / `singleton` は cache に各フラグを単独追加、`all` は両方を追加。CPU では host と device は同じ CPU メモリなので配置の利益は測れない。

| 条件 | Small（2根） | C相当（4根） | C相当の速度比 |
|---|---:|---:|---:|
| before | 0.026526 | 0.994918 | 1.000x |
| host | 0.026353 | 0.998221 | 0.997x |
| cache | 0.026322 | 0.998716 | 0.996x |
| forced | 0.023844 | 0.980861 | 1.014x |
| singleton | 0.021127 | 0.695868 | 1.430x |
| all | 0.018743 | 0.687719 | 1.447x |

各行は単独の比較で、短縮量は足し合わせない。host/cache の差はCPUでは測定の揺れ程度。

## 段階別（秒 / 探索呼び出し）

| 規模・条件 | サンプル | 候補 | ロールアウト | 葉 | 合計 |
|---|---:|---:|---:|---:|---:|
| Small before | 0.000358 | 0.004941 | 0.014056 | 0.006946 | 0.026526 |
| Small all | 0.000341 | 0.004677 | 0.008743 | 0.004755 | 0.018743 |
| C相当 before | 0.006189 | 0.024565 | 0.735300 | 0.228419 | 0.994918 |
| C相当 all | 0.006154 | 0.024309 | 0.492319 | 0.164736 | 0.687719 |

各段階も独立した中央値なので、和は合計の中央値と必ずしも一致しない。1根あたりは Small を2、C相当を4で割る。C相当の合計は1根あたり 0.248729 → 0.171930 秒。

C相当の host 条件で workspace 検索/再利用は約0.10 ms、load_worlds は約12.5 ms（呼び出し合計の約1.3%）。既存の再利用を維持し、C++ の変更は行わなかった。これらの時間は候補・ロールアウトにも含まれる内数。

## 結果の一致とフラグ

- `--skip-forced-inference`: 強制する根の行を方策推論から除外。次の決定は substep0 から encode する。RNG の消費順が変わる。デフォルト無効。
- `--skip-single-candidate`: 根の完全な候補手を各世界で実行して合法性を確認した後、続きを省略する。全世界で不適合なら従来どおり fallback。選択確率は1。RNG の消費順が変わる。デフォルト無効。
- 省略した q は `null`、report に `rollout_skipped: single_candidate`。`singleton_roots` / `singleton_validated_worlds` / `rollout_roots` / `rollout_worlds` / `forced_inference_rows_skipped` で区別。`report_values=True` は q の診断を優先し単一候補でも通常ロールアウトする。
- `--search-cache-device auto|host|device` と `--search-cache-mb`（既定256 MiB）: 分割推論時の token cache の配置のみ変更。容量超過なら host に置く。auto/CUDA は空きメモリの1/4も上限とする。推論 batch と演算順・dtype は変更しない。max-inference-batch0 は既存の通常 Actor 経路を維持。
- キャッシュ重複行のコピーを NumPy/Torch の gather/copy にまとめた。配置変更とともにビット一致を期待する変更。CPU の before/host/cache は候補、全 q、確率、選択手が厳密一致。
- `--root-batch-size N`（既定0）: 根を最大N個ずつ処理し、SearchEnv と belief / token cache のメモリを制限。独立な探索の RNG の割当順が変わるため任意フラグ。通常 Actor による実対局の推論 batch は変更しない。
- `seconds_per_search` は従来どおり1根あたり。`seconds_per_call` と `roots_per_search_call` は実対局の探索呼び出し単位。`search_batches` / `max_batch_roots` は分割後の workspace 単位。

完全な従来経路: `--no-skip-forced-inference --no-skip-single-candidate --root-batch-size 0 --search-cache-device host`。auto の配置・コピー変更も CPU では厳密一致した。

既存 `search_before.json` に加え、変更前実装で `search_exact_before.json` を作成。4対局の全手順、候補、log_pi、log_rho、q、探索確率、選択手、対局集計を許容誤差なしで照合（host / auto / device）。根の推論省略は (world row, decision, substep) に対応付けたカテゴリ乱数による coupling で、全世界の葉値・q・探索分布の厳密一致を D0/D3、分割なし/5行で検証。単一候補は固定根400回の方策周辺分布との比較、全/部分世界の不適合、診断モードを検証。

通常 Actor、学習、収集、既存評価、goodfield、DESIGN.md、docs/、C++ は未変更。依存追加・commit はなし。

## 再計測

```sh
baseline=$(mktemp -d /tmp/gfrl-search-before.XXXXXX)
git show 8459231:python/gfrl/search.py > "$baseline/search.py"
git show 8459231:python/gfrl/actor.py > "$baseline/actor.py"
.venv/bin/python scripts/bench_search.py --micro \
  --baseline-dir "$baseline" --repeats 5 \
  --checkpoint runs/m4-s-ppo-v1/checkpoint.pt \
  --belief b1 --fit runs/belief-smoke/fit.pt \
  --roots 4 --root-steps 20 --worlds 64 --depth 3 \
  --candidates 16 --candidate-samples 64 --max-inference-batch 8192 \
  --out runs/search-optimization/c-final.json
```

Small は `--roots 2 --root-steps 0 --worlds 8 --depth 1 --candidate-samples 16` に変更。MPS が利用可能な環境では同じコマンドに `--device mps` を追加。CUDA の内訳が必要な場合は `--profile` を追加できるが同期の overhead が増えるため、通常の速度比較とは分ける。

TSUBAME では M の POLICY と FIT を指定してまず同じ micro benchmark (`--device cuda --roots 8 --search-cache-mb 1024`) で各変更の寄与と before/host/cache の厳密一致、`peak_cuda_bytes` を確認する。

実対局での設定 C・環境数比較（下記の flags は明示的に最適化を有効にする）:

```sh
for cap in 8 0; do
  for n in 128 512 1024; do
    python scripts/eval_search.py \
      --checkpoint "$POLICY" --belief b1 --fit "$FIT" \
      --device cuda --threads 1 --seed 20261109 --pairs 1000 \
      --num-envs "$n" --worlds 64 --depth 3 \
      --candidates 16 --candidate-samples 64 --beta 0.1 \
      --leaf-view actor --max-inference-batch 8192 \
      --skip-forced-inference --skip-single-candidate \
      --search-cache-device auto --search-cache-mb 1024 \
      --root-batch-size "$cap" \
      --out "runs/search-C-n${n}-cap${cap}.json"
  done
done
```

cap8 は workspace を制限する条件、cap0 は分割なしの比較条件。cap0 がメモリ不足なら cap8 の結果を使用する。速度だけの一次比較は pairs を100程度に下げられる。JSON の `roots_per_search_call`、`seconds_per_call`、`seconds_per_search`、`candidate_paths / batch_roots`、`singleton_roots / batch_roots`、`cache_cpu_actors` / `cache_cuda_actors` / `cache_peak_bytes` を併記する。従来条件との比較には省略2フラグを無効、cache をhost、cap0にする。

## 環境数の比較（CPU、C相当）

20 stage 後の固定スナップショットで対応可能な全根を探索。省略2フラグ有効、root-batch-size8、warmup後1回。対局全体の速度ではなく探索1呼び出しの計測であり、異なる環境数の局面集合も異なる。

| num-envs | 呼び出しの根数 | 合計 秒/呼び出し | 秒/根 | 平均候補数 | 単一候補率 | 最大workspace根数 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 24 | 6.793310 | 0.283055 | 2.208 | 29.2% | 8 |
| 512 | 121 | 32.204514 | 0.266153 | 2.149 | 38.8% | 8 |
| 1024 | 252 | 73.407112 | 0.291298 | 2.298 | 32.1% | 8 |

根数を増やしても CPU の1根あたりは約0.27–0.29秒で、CPUでは明確な追加高速化はない。workspace の根数は8に制限でき、1024環境も完走した。GPUの並列性による変化はTSUBAMEでの比較が必要。

再現用（上記 baseline の準備後）:

```sh
for n in 128 512 1024; do
  .venv/bin/python scripts/bench_search.py --micro \
    --baseline-dir "$baseline" --repeats 1 \
    --checkpoint runs/m4-s-ppo-v1/checkpoint.pt \
    --belief b1 --fit runs/belief-smoke/fit.pt \
    --roots 0 --num-envs "$n" --root-steps 20 \
    --worlds 64 --depth 3 --root-batch-size 8 --variants bounded \
    --out "runs/search-optimization/scale-$n.json"
done
```

## 検証結果

- `.venv/bin/cmake --build build -j 4`: 成功（Release、差分なし）。
- `.venv/bin/ctest --test-dir build --output-on-failure`: 17/17 passed。
- `.venv/bin/python -m pytest -q`: 155 passed, 1 skipped（50.09秒）。既存の非 writable NumPy view に関する PyTorch warning が1件。
- S checkpoint を用いた paired 4対局の before/after: 全手順・候補・q・事前分布・選択手・対局集計一致、最大 q 誤差0。
- 省略2フラグ・root-batch-size2を有効にした S checkpoint の4対局: 完走、real_mismatches=0。
- デバイス cache の実演算は CPU のみ検証。CUDA 空き容量による配置分岐は mock で検証したが、GPU 実測の代わりにはならない。
