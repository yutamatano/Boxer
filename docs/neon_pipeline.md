# Neonの解析とRemote SSHでの結果確認

## 実行順序

`prepare_neon_vio.py` → `mono_inertial_neon` →
`convert_neon_trajectory.py` → `run_boxer.py` → 動画・CSV確認です。
`run_neon_pipeline.py` はこの順序で全処理を実行します。

初回またはC++コード更新後は `bash scripts/build_neon_vio.sh` を実行してください。
Neon reader、SciPy、PyTorch、Boxerの重み、FFmpegが必要です。

```bash
cd /home/ubint-lab/boxer
source .venv/bin/activate

python run_neon_pipeline.py \
  --input neon_import_1 \
  --output output/neon_run_01 \
  --max_n 618 \
  --width 800 \
  --imu-offset-ms 60 \
  --experimental-noise \
  --fuse
```

`--input neon_import_1` は `sample_data/neon_import_1` の直下にある単一の
記録フォルダも解決します。複数の記録がある場合は対象フォルダを直接指定します。
出力先は空である必要があります。再試行では新しい名前を使ってください。

上記の+60 msは今回の記録用の**試験設定**です。Neon一般の既定値ではありません。
既定の時刻補正は0 msです。校正済みの値があれば `--imu-offset-ms`、
`--imu-noise noise.json`、`--extrinsics extrinsics.json` で指定してください。
`--experimental-noise` は未校正のノイズ設定を明示的に許可します。

`--vio-only` で軌跡と点群までを実行できます。保存済みの結果からBoxerだけを
実行する場合は同じ出力先に `--boxer-only` を指定します。

```bash
python run_neon_pipeline.py --input neon_import_1 \
  --output output/neon_run_01 --boxer-only --fuse
```

`--prepared PATH` を指定すると既存の前処理を再利用します。その場合、
フレーム範囲・解像度・時刻補正・IMU設定は前処理側の設定が使われます。
`--labels chair,table,monitor` で対象物体を限定できます。

## 出力と区間の扱い

```text
output/neon_run_01/
  prepared/                  補正画像、IMU、同期・画像品質診断
  session_000/slam/          states.csv、camera_trajectory.csv、map_points.csv
  session_000/geometry/      geometry_segment_*.npz、report.json
  session_001/...            追跡失敗後の再開セッション
  boxer_000/<recording>/     3D検出・追跡・融合CSV、MP4
  pipeline.json             有効区間、点群数、カバー率、結果パス
```

マップリセット前に初期化済みの区間を保存し、残りの画像は別プロセスで
再開します。保存時にはSLAMのバックグラウンド最適化の終了を待ちます。
変換は実行時と最終マップの両方でIMU初期化済み、かつ追跡成功のフレームだけを
採用します。未初期化・追跡失敗フレームを補間して埋めることはありません。
区間ごとに原点が異なるので、独立したBoxer処理として出力します。

疎点群はSLAMが三角測量した点であり、深度センサーの測定ではありません。
無効な点を除き、3キーフレーム以上で観測された点を出力します。
カメラ姿勢と点群を同じ量だけ平行移動し、同じメートル単位のZ-up座標系に
揃えます。Boxerは利用可能な点群を深度入力として使います。
静的マップの点を「現在のフレームで見えている」という証拠には使いません。

主なファイルは `boxer_3dbbs.csv`（各フレームの検出）、
`boxer_3dbbs_tracked.csv`（追跡履歴）、`boxer_3dbbs_fused.csv`
（`--fuse`指定時の静的物体）、`boxer_viz_final.mp4` です。
追跡と融合は別々の処理として順番に実行します。

## 原因調査と残る制約

2026-10-10の `neon_import_1` は618フレーム、約61.74秒、約10 fpsです。

- 旧実装では初期化後の追跡失敗でマップがリセットされ、最後にだけ軌跡を
  保存するため、それ以前の有効な軌跡も失われました。空のCSVを変換する
  際のエラーはこの結果です。最小区間長を小さくしても解決しません。
- メーカー公称のカメラ–IMU変換、ジャイロのdeg/s→rad/s、
  加速度のg→m/s²は公式資料と整合します。
  [座標変換](https://docs.pupil-labs.com/alpha-lab/imu-transformations/)、
  [データ単位](https://docs.pupil-labs.com/neon/data-collection/data-format/)。
- 画像とジャイロの回転比較では時刻補正の候補が区間ごとに変わりました。
  +60 msは今回比較した設定の中で有効区間を得られましたが、校正値とは
  断定できません。前処理は独立した3つの時間区間の同期候補も記録します。
- 追跡を失うフレーム348付近ではぼけと白飛びを確認しました。
  800幅の補正画像で明度250超の割合は約41%、Laplacian分散は約34でした。
  これらは画像品質の観測値であり、失敗原因を単独で証明するものではありません。
  前処理の `frames.csv` に画像品質指標を記録します。
- 幅1200・特徴点3000の比較では有効区間を得られず、+40 msの試験では
  69フレームでした。幅800・特徴点2000・+60 msでは117フレームを保存できました。
  SLAMのスレッドスケジューリングにより境界が変わることがあります。

これらの修正は全618フレームの連続した軌跡や実寸精度を保証しません。
記録時のぼけ・白飛びはコードで元に戻せず、IMUノイズと同期・外部パラメータの
校正、外部基準軌跡による評価が残ります。点群付き3D物体推定の出力成功と、
位置・サイズの精度検証は区別してください。

## VS Code Remote SSHから動画を開く

Remote SSHのターミナルは研究室PC上で動作します。
`xdg-open` や `ffplay` で手元のPCの画面を直接開くことはできません。

最も簡単なのは、VS Codeのエクスプローラーから `boxer_viz_final.mp4` を
ダウンロードして手元のPCで開く方法です。

ブラウザで開く場合は、研究室PC側で次を実行します。

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory output/neon_run_01
```

VS Codeの「ポート」タブで8000を転送し、表示されたローカルアドレスを
手元のブラウザで開きます。`boxer_000/<recording>/boxer_viz_final.mp4` を
選択してください。使用後はターミナルのCtrl+Cでサーバーを止めます。

動画は最大1920×1080内に縮小したH.264・yuv420p・faststartで保存し、
解像度とfpsに合ったレベルをエンコーダーが選びます。
生成失敗を検出し、既存の完成動画を途中ファイルで置き換えないようにしています。
