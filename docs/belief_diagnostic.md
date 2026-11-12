# 未使用の相手手札に対するオフライン診断

```sh
.venv/bin/python scripts/belief_diagnostic.py --device mps \
  --out runs/belief-diagnostic.json
```

既定値は現方策 `runs/play/m-2166.pt`、AR `runs/play/belief-ar-m.pt`、
5,000局面、各64サンプル、採取確率0.1、pool fraction 0です。
`--policy runs/belief/policy.pt` などで方策ドリフトを調べられます。
両方を比較する際は同じseed・バッチ設定を使用してください。
比較方策ごとに自動fitすれば、その方策に適合したB1との比較になります。
同じ `--fit` を指定すればB1を固定できます。

`--fit` は `scripts/make_belief_data.py --labels-only` が生成するtorchファイルです。
探索と同じ `fit_baselines` により、**公開使用カードを含むretainedラベル**からB1をfitします。
省略時は、指定方策による別の自己対戦50,000局面（seed+1）をlabels-onlyで生成します。
評価局面はseed、fit局面はseed+1で、サンプル生成のTorch RNGも方策のRNGから隔離します。
出力JSONにfit元と生成設定を記録します。

出力はJSON、同じstemのMarkdown、`.positions.pt` の3ファイルです。
最後のファイルには観測、元ラベル、真の未使用手札・枚数、game ID、決定番号、
終局までの決定数、方式ごとの受理サンプル数・局面別指標とECE用のbin統計を保存します。
特権情報を含むので、学習・公式評価の入力には使用しません。

## 比較方法

AR、Hybrid、B1は実際の `SearchAgent._sample_worlds` を呼びます。
有効なassignmentを再度 `env.unused_assignment` に渡した返り値だけを採点します。
ARの生成結果に公開使用カードが足りない場合、探索の変換は各カードで0に打ち止めにします。
正解の抽出では `label_opp_hand − public used` を厳密に計算し、非負をassertします。
Hybridは探索の実装どおりARのcountからB1の内容を生成し、公開使用カードを加えて検証します。
B1も探索どおりretained assignmentを生成してから使用カードを減算します。

`uninformed` はB1の内容分布から**真の未使用枚数**だけ引き、公開使用カードを加えて
`unused_assignment` で検証する特権的な参照方式です。
受理条件には非giftカードや手札容量なども含まれるため、真の枚数を与えても棄却は起こり得ます。
各スロットの上限は探索と同じ100試行です（`--max-attempts` で変更可）。
JSONは棄却数・棄却率、上限到達スロット、サンプル0の局面数も報告します。
サンプル0の局面は指標から除外し、部分受理は実際の受理数を分母にします。

presenceはpaddingを除く296種類で計算します。BrierとECEは生の頻度、
log-lossだけ `(presence回数+0.5)/(受理数+1)` を使います。
active指標は、平滑化前に「正解あり、またはサンプルに出現あり」の種類に絞ります。
Brier、log-loss、copies MSEは種類について合計してから局面平均します。
枚数modeの同点は小さい枚数を選びます。
ECEは10binで局面×カードをpoolして計算します。
95% CIはゲーム全体を復元抽出するpercentile bootstrapです（既定1,000回）。
局面平均の重みは維持するので、ゲームを同じ重みで平均する指標とは異なります。

残り決定数は「終局のengine decision数 − 現在の0始まりdecision index」で、
現在の決定を含みます。自動決定も含み、採取済みゲームを最後まで進めて確定します。
区間は `[0,5), [5,10), [10,20), [20,30), [30,45), [45,inf)` です。
カテゴリは既存のcard featureのcategory one-hotから分けます。
カテゴリ別のsize指標は手札全体の値を再掲し、カード指標だけそのカテゴリに絞ります。

小規模動作確認:

```sh
.venv/bin/python scripts/belief_diagnostic.py --device cpu \
  --positions 40 --samples 8 --fit-positions 100 --num-envs 4 \
  --bootstrap 100 --out /tmp/belief-smoke.json
```

このfit数は動作確認用です。精度比較では十分なfit局面数を使用してください。
CPU/MPS/CUDAに対応し、新規依存・ネットワークアクセスは不要です。

## CPU計測とMPS所要時間の目安

この環境で現行チェックポイントを使った計測:

| 設定 | 総時間 | fit | 採取・サンプリング・終局待ち |
|---|---:|---:|---:|
| 40局面×8、fit 100、4 env、bootstrap 100 | 11.1秒 | 4.8秒 | 6.0秒 |
| 80局面×64、fit 1,000、64 env、bootstrap 100 | 118.6秒 | 43.6秒 | 74.7秒 |

後者のサンプリング内訳はAR 13.8秒、Hybrid 6.1秒、B1 0.21秒、
uninformed 0.16秒でした。pytestとの並行実行による負荷を含みます。
また、少数局面では終局までの追加自己対戦が相対的に大きいため、
総時間をそのまま5,000/80倍する見積もりは過大になります。

粗いCPU換算ではfit 50,000局面で約36分、評価5,000局面のサンプリングで約21分、
これに採取・終局待ち・bootstrapが加わります。
MPSでモデル推論部分がCPUの3〜10倍になると仮定すると、既定の自動fit込みで
**およそ10〜30分**を初回の実行枠の目安にしてください。
これはMPS実測ではありません。Appleチップ、試合長、棄却率によって大きく変わり、
カード生成・検証・bootstrapなどCPUで行う処理は同じ比率では高速化しません。
`--fit` の既存ファイルを使えばfit分を省けます。
正確な見積もりにはMPS上でfit 1,000・評価80程度を先に実行し、JSONのtimingsを参照します。
