# Neon視線とTag Alignerの接続

`--neon_gaze` を指定すると、Neonの計測済み視線を画像に重ね、
`boxer_gaze.csv` に保存します。Tag Alignerの `aligned_poses.csv` を
取り込むと、Boxerの3D物体と視線の方向を同じメートル単位・Z-upの
実環境座標で出力できます。Boxer自身が眼球画像から視線を推定する処理ではありません。

## 1. 手元のNeonで視線表示を確認

リポジトリ直下で、既存のBoxer環境を使用します。
Neon readerとSciPyが必要です（既存のNeon対応と同じ依存関係です）。

```bash
boxer/bin/python run_boxer.py \
  --input sample_data/2026-10-10_09-39-40-d4fe821f-20261015T025507Z-1-001/2026-10-10_09-39-40-d4fe821f \
  --start_n 0 --max_n 90 --force_cpu --point_cloud off \
  --labels chair,table,monitor --neon_gaze \
  --output_dir output/neon_gaze
```

記録名のサブディレクトリに、視線を黄色で重ねた `boxer_viz_final.mp4`、
`boxer_gaze.csv`、座標系と処理条件を記録した `boxer_input.json` が保存されます。
姿勢を指定しない場合、CSVは画像座標とカメラ座標の視線方向までを保存します。
この場合の世界座標欄は空欄です。

## 2. 実環境へ整列するための入力

[Tag Alignerの公式手順](https://docs.pupil-labs.com/alpha-lab/tag-aligner/)に従い、
同じ記録の `aligned_poses.csv` を用意します。

- Pupil CloudのReference Image Mapper（RIM）から得たカメラ姿勢。
- 同じRIM再構成に含まれる記録に写った、静止したtag36h11 AprilTag。
- タグの黒い部分の一辺の実寸、中心位置、向きの実測値。
- これらを使ってTag Alignerで整列した `aligned_poses.csv`。

詳しいタグ設定と実行方法は[公式README](https://github.com/pupil-labs/tag-aligner#usage)を参照してください。
Boxer側の入口は**整列済みCSV**です。RIM処理、タグの検出・実測、
`alignment.json` の算出は公式Tag Aligner側で行います。
生の `poses.p` や別記録のCSVをそのまま渡すことはできません。

今回確認した `sample_data` の4件のNeon記録には視線データがありますが、
`aligned_poses.csv` とタグの実測情報はありません。
したがって、これらの記録を実測した実環境座標へ整列した結果はまだありません。
タグの位置・大きさを仮定して校正済みとして扱わないでください。

## 3. 整列済み姿勢をBoxer用に変換

同じ記録の `info.json` の `start_time` と、元のシーンカメラ `.time` を使います。
CSVを例えば `sample_data/<記録名>/aligned_poses.csv` に置いて実行します。

```bash
boxer/bin/python convert_tag_aligner.py \
  --input sample_data/<記録名> \
  --aligned-poses sample_data/<記録名>/aligned_poses.csv \
  --meters-per-unit 1 \
  --output output/tag_alignment
```

`--meters-per-unit` は必須です。タグ実測値の単位がmなら `1`、mmなら
`0.001` を指定します。これは単位換算であり、Tag Alignerの整列スケールを
もう一度適用する処理ではありません。

Tag Alignerの世界座標 `(x, y, z)` をBoxerの `(x, z, -y)` へ変換します。
Tag Alignerの基準世界はXが右、Yが下、Zが前の公式規約で設定してください。
世界座標を回転し、カメラ自身のOpenCV軸（右・下・前）は維持します。
原点はタグ基準で設定した原点を維持します。

出力先は空のディレクトリを指定します。
`report.json` に採用フレーム数、欠測数、各区間の `start_n` と `max_n`、
`geometry_segment_*.npz` の名前を記録します。
CSVの `[start_timestamp, end_timestamp)` 内のフレームのみ採用し、
欠けた区間は補間・外挿せず別のセグメントに分けます。
時刻が合わなくなるため、動画の先頭を時刻0に置き換えないでください。

## 4. 同じ座標系で物体と視線を出力

`report.json` にある区間ごとに実行します。以下の `START` と `COUNT` を
その区間の `start_n` と `max_n` の整数に置き換えてください。

```bash
boxer/bin/python run_boxer.py \
  --input sample_data/<記録名> \
  --neon_geometry output/tag_alignment/geometry_segment_000.npz \
  --start_n START --max_n COUNT --skip_n 1 \
  --neon_gaze --point_cloud off --track --force_cpu \
  --labels chair,table,monitor \
  --output_dir output/neon_aligned_000
```

選択フレームに対応する正確な姿勢がない場合はエラーになります。
各区間は別の出力先で処理します。元の物体CSV・追跡履歴の形式はそのままです。
別のVIO/SLAMマップの点群を混ぜる場合、その点群も同じ実環境座標へ整列する
必要があります。この変換コマンドは点群や3Dモデルを生成しません。

## CSVと精度上の意味

`boxer_gaze.csv` は**選択した映像フレームごとに1行**です。
元の200 Hz等の全視線サンプルを保存する形式ではありません。
検出物体が0件でも行を保存します。

| 列 | 意味 |
| --- | --- |
| `frame_id`, `time_ns` | 元の映像フレーム番号とint64 ns時刻 |
| `gaze_time_ns`, `gaze_delta_ns` | 採用した視線時刻と「視線－映像」の時刻差 |
| `raw_x/y` | 元の歪んだシーン画像上のNeon視線pixel |
| `image_x/y`, `in_image` | 歪み補正・リサイズ後の座標と画像内かどうか |
| `direction_camera_x/y/z` | カメラ座標での単位視線方向 |
| `origin_world_x/y/z` | シーンカメラ中心の位置（m） |
| `direction_world_x/y/z` | 同じBoxer世界座標の単位視線方向 |
| `status` | `missing_gaze`, `invalid_gaze`, `outside_scene`, `camera_only`, `world_ray` |

最も近い視線を採用し、既定では映像時刻との差が10 msを超えると欠測です。
`--gaze_max_delta_ms` で許容差を明示的に変更できます。
左右の眼球の光軸を注視方向と取り違えず、Neon readerのbinocular gazeを使います。
歪みを逆変換したカメラ光線を姿勢の回転で世界へ変換します。

`origin_world + distance * direction_world` は視線方向に沿った光線です。
原点は**シーンカメラ中心で近似**しており、眼球中心ではありません。
実測の注視深度、物体との交点、注視対象IDはこのCSVからは確定しません。
近距離の視差、視線・姿勢の誤差、フレーム内の時間差が残ります。
`world_ray` は座標変換ができたという意味であり、絶対位置精度を保証しません。
Tag Aligner姿勢の有無と由来は `boxer_input.json` の `alignment` で確認できます。
点群なしのBoxerによる物体の深度・サイズもモデルの推定です。

`--skip_viz` は画像出力だけを、`--no_csv` はCSV/メタデータ出力を無効にします。
`--cache3d` はフレーム読込を省略するため `--neon_gaze` と併用できません。
現在の表示先は `run_boxer.py` のJPEG/MP4です。
既存の対話型 `view_tracker.py` はNeon入力に未対応です。

## 検証

```bash
boxer/bin/python -m pytest -q tests/test_neon_gaze.py tests/test_neon_loader.py \
  --basetemp=output/neon_gaze_tests
```

合成データで既知の回転・並進・単位変換、歪み補正、ナノ秒精度、
欠測区間とCSVの保存を検証します。実Neonの表示・出力確認と、
実測タグを使った絶対位置精度の評価は区別してください。

2026-10-15の確認では、`2026-10-10_09-39-40-d4fe821f` の618フレームで
視線CSVを出力し、最大時刻差は8.573 msでした。
同じ記録のフレーム10～12をCPU・点群なし・`chair,table,monitor` で比較し、
視線オプションの有無で2D検出CSV（14件）と3D検出CSV（13件）が
バイト単位で一致しました。位置精度や処理速度の改善を示す比較ではありません。
