# COLMAP単独のNeon復元実験

この実装は `experiment/neon-colmap-standalone` ブランチの試験コードです。
画像・メーカーのカメラ校正・元の画像時刻だけを使います。
ORB-SLAM3、IMU、既存のVIO軌跡、姿勢事前分布は復元に使いません。

## 導入・実行

BoxerのPython環境と分けてください。研究室PCで検証したバージョンは
CUDA版PyCOLMAP 4.2.1です。CPU版とCUDA版は同じPythonモジュール名を使うため、
両方を同じ環境へインストールしないでください。

```bash
uv venv output/colmap_env --python .venv/bin/python
uv pip install --python output/colmap_env/bin/python \
  'pycolmap-cuda12==4.2.1' numpy opencv-python scipy \
  pupil-labs-neon-recording matplotlib

output/colmap_env/bin/python run_neon_colmap.py \
  --input neon_import_1 \
  --output output/neon_colmap_run_01 \
  --max_n 618 --skip_n 1 --width 1000 \
  --matching hybrid --mapper global
```

CPU環境では `scripts/requirements-colmap.txt` をインストールし、実行時に
`--cpu` を追加します。出力先は空である必要があります。

`--matching sequential` は近接・間隔を広げた画像ペアを照合します。
`exhaustive` は全ペア、`hybrid` は近接ペアに加えて3画像おきの全ペアを照合します。
`--loop-stride` で後者の間隔を変えられます。
`--mapper incremental` と `global` はCOLMAP内の復元方式の比較です。
グローバル方式も外部の軌跡を使いません。

```bash
# 同じ特徴点・対応点・評価分割で復元方式を比較する場合
output/colmap_env/bin/python run_neon_colmap.py \
  --input neon_import_1 --output output/neon_colmap_repeat \
  --matching hybrid --mapper global --reuse-data output/neon_colmap_run_01
```

再利用は画像・データベース・評価分割に限定し、前の復元姿勢は読み込みません。
元のデータベースをコピーし、出力を上書きしません。

## 評価方法

選択画像の5枚に1枚を先に評価用へ分け、SfMの登録画像から除外します。
登録後にも評価画像がモデルへ混入していないことを確認します。
カメラ内部パラメータはメーカー校正に固定します。OpenCVで歪み補正し、
COLMAPの画素中心規約に合わせて主点へ0.5 pixelを加算します。

評価画像を学習画像の3D点と照合し、対応点の70%だけを使ってPnPで姿勢を
推定します。残り30%は姿勢推定・姿勢微調整に使わず再投影誤差を計算します。
30対応点未満または姿勢推定に失敗した画像も成功率の分母に含めます。
4 pixel以内の割合には失敗投影を含め、大きな外れ値を除外していません。

これは幾何検証済みの画像対応に対する**学習外の視覚整合性**です。
独立した基準軌跡に対するカメラ位置誤差や、物体の寸法誤差ではありません。
実測基準がないため、cm単位の精度は報告できません。

## 実データ試験結果

入力は `neon_import_1/2026-10-10_09-39-40-d4fe821f` の618フレームです。
最終試験は全画像を1000×750に補正し、学習495画像・評価123画像に分けました。

| 条件 | 学習画像の登録数 | 最大モデル | モデル数 |
|---|---:|---:|---:|
| 3フレーム間隔、近接照合、逐次復元（初期動作確認） | 3 / 165 | 3画像・7点 | 1 |
| 3フレーム間隔、全ペア照合、逐次復元 | 5 / 165 | 5画像・111点 | 1 |
| 同じ全ペア照合、グローバル復元 | 132 / 165 | 51画像・868点 | 11 |
| 全フレーム、hybrid照合、グローバル復元（最終再実行） | **469 / 495** | **252画像・4,398点** | **9** |

初期動作確認の後、画素中心規約と照合時の特徴点数上限を修正しています。
最初の行は同一設定での方式比較ではありません。

最終試験では、分断したモデル全体で学習画像の94.7%を登録しましたが、
一つの座標系につながった最大モデルは学習画像の50.9%でした。
モデルの和集合を「一つの部屋全体の復元」と解釈してはいけません。

最大モデルの評価結果は次のとおりです。

| 指標 | 値 |
|---|---:|
| 学習画像の再投影誤差中央値 | 1.01 px |
| 学習画像の再投影誤差95%値 | 3.58 px |
| 最大モデルで位置推定できた評価画像 | 43 / 123 |
| 学習外評価点数 | 1,791 |
| 学習外再投影誤差中央値 | **2.25 px** |
| 学習外再投影誤差95%値 | **10.69 px** |
| 学習外再投影RMSE | **38.71 px** |
| 学習外評価点の4 px以内の割合 | **72.5%** |
| いずれかのモデルで位置推定できた評価画像 | 70 / 123（56.9%） |

RMSEが大きいことから、一部に大きな誤対応・推定誤差が残っています。
画像内の低い学習誤差だけを根拠に、高精度な部屋モデルとは判断できません。
同じデータベースからの再実行でも、数値最適化・スレッドの影響で点数や誤差は
少し変わります。先行実行は最大4,273点、学習外中央値2.38 px・RMSE36.74 pxでした。
登録469画像・最大252画像・9モデルという範囲は再現しました。

今回得られたのは**分断した疎点群**です。密な点群・メッシュは作っていません。
部屋全体が一つにつながった復元や絶対的な実寸精度は未達成です。

## 出力と確認

```text
output/neon_colmap_final_05/
  frames.csv                   元のframe_id・整数ナノ秒時刻・train/holdout
  manifest.json                入力・校正・非メートル単位の明記
  database.db                  特徴点・画像対応
  report.json                  全モデルと評価画像ごとの測定値
  sparse/0/points.ply           最大モデルの色付き疎点群
  sparse/0/reconstruction.png   点群とカメラ位置の確認図
  sparse/0/camera_poses.npz     登録学習画像だけのカメラ姿勢
```

`camera_poses.npz` は `T_reconstruction_camera` を保存します。
スケールは任意で、重力方向にも合わせていません。
メートル単位の `T_world_camera` としてBoxerへ直接渡すことはできません。
未登録画像や評価用に推定した姿勢を、正式な復元姿勢として補間・出力しません。
異なるモデルの点群をそのまま結合することもできません。

Remote SSHではPNG・PLYをVS Codeからダウンロードして確認できます。
画像だけの再構成で時間同期やIMUの問題を切り離せましたが、ぼけ・白飛び・
視差不足、反復模様の誤対応などの影響は残っています。

仕様確認には[PyCOLMAP公式API](https://colmap.github.io/pycolmap/index.html)と
[COLMAP公式チュートリアル](https://colmap.github.io/tutorial.html)を使用しました。
