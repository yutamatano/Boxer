# Neon視線とTag Alignerの接続

`--neon_gaze` を指定すると、Neonの計測済み視線を画像に重ね、
`boxer_gaze.csv` に保存します。Tag Alignerの `aligned_poses.csv` を
取り込むと、Boxerの3D物体と視線の方向を同じメートル単位・Z-upの
実環境座標で出力できます。Boxer自身が眼球画像から視線を推定する処理ではありません。

`--gaze_targets` を指定すると、視線とBoxerの3Dボックスの交点から
**注視位置の候補**まで推定します。点群なし・軌跡なしでもフレームごとに動作します。
`--gaze_targets` は `--neon_gaze` も有効にします。

## 1. 手元のNeonで視線表示を確認

リポジトリ直下で、既存のBoxer環境を使用します。
Neon readerとSciPyが必要です（既存のNeon対応と同じ依存関係です）。

```bash
boxer/bin/python run_boxer.py \
  --input sample_data/2026-10-10_09-39-40-d4fe821f-20261015T025507Z-1-001/2026-10-10_09-39-40-d4fe821f \
  --start_n 0 --max_n 90 --force_cpu --point_cloud off \
  --labels 'chair,table,monitor,laptop,cell phone' --gaze_targets \
  --output_dir output/neon_gaze
```

記録名のサブディレクトリに、視線を黄色で重ねた `boxer_viz_final.mp4`、
`boxer_gaze.csv`、座標系と処理条件を記録した `boxer_input.json` が保存されます。
姿勢を指定しない場合、CSVは画像座標とカメラ座標の視線方向までを保存します。
この場合の世界座標欄は空欄です。

### 点群なしで注視位置を推定する

上記コマンドでは、追加で `boxer_gaze_targets.csv` を保存します。
視線方向と、そのフレームで検出した3Dボックスの表面との交点を計算し、
手前から最初の交点を注視位置の候補にします。
動画には対象名と推定距離を表示し、対象ボックスを緑色で強調します。
対象として検出する物体を `--labels` に含めてください。

| 列 | 意味 |
| --- | --- |
| `frame_id`, `time_ns`, `gaze_time_ns`, `gaze_delta_ns` | 映像・視線の元の時刻と対応 |
| `status` | `estimated` の場合のみ位置を採用。欠測の理由も保存 |
| `method` | `predicted_obb_surface`（推定ボックス表面との交点） |
| `coordinate_frame` | `position_frame_*` と物体CSVが使う座標系 |
| `point_cloud_used` | このフレームのBoxer入力に点群を使ったか |
| `box_count`, `intersection_count` | 検出ボックス数と視線が交差したボックス数 |
| `detection_index`, `label`, `box_confidence` | 現在の検出の添字、対象名、物体の信頼度 |
| `camera_inside_box` | カメラがボックス内ならtrue。前方の出口との交点を採用 |
| `distance_m` | カメラ中心から交点までの推定距離 |
| `position_camera_x/y/z` | カメラ座標（右・下・前）での推定位置 |
| `position_frame_x/y/z` | 同じ時刻の物体CSVと同じ座標系での推定位置 |

軌跡がない場合の `coordinate_frame` は `per_frame_assumed_upright` です。
各フレームの原点はカメラ中心であり、フレーム間の位置比較には使えません。
SLAM3由来の有効な軌跡NPZを `--neon_geometry` に指定すると
`world_z_up` となり、そのSLAM区間の座標系で出力します。
同じ仕組みでTag Alignerの整列済み軌跡も利用できます。

視線がない・画像外・物体がない・交点がない場合は位置欄を空欄にします。
それぞれ `missing_gaze` / `invalid_gaze` / `outside_scene` / `outside_image` /
`no_boxes` / `no_intersection` を記録し、前フレームの位置で埋めません。
`detection_index` は同じ `time_ns` の3D検出CSVの0始まりの行番号です。
Track IDや、時間的に安定した物体IDではありません。

点群切り替えは従来どおりです。

- `--point_cloud off`：点群を入力せず、Boxerが予測した奥行き・サイズを使用。
- `--point_cloud auto`：登録済みの点群があればBoxerの入力に使用。
- `--point_cloud on`：登録済み点群が必須。欠測はエラー。

交点計算自体はどのモードでも3Dボックスを使います。
将来LiDARを使う場合も、この実装ではLiDAR点に直接当てる処理は行いません。
カメラ姿勢と同じ座標系へ整列した点群を、既存の `points_world` として
入力すると、点群を使ったBoxerの推定結果から交点を求めます。

点群なしの奥行きとサイズは学習モデルの推定値で、交点はボックス表面の
近似です。実測した物体表面の位置や絶対距離の精度は保証されません。
また、時間的な固視・サッカードの分類は行いません。
`box_confidence` は物体検出の信頼度で、注視判定の確率ではありません。

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

姿勢NPZを指定すると、対応するフレームを自動判定します。
`--start_n 0` で記録全体を解析でき、区間の開始・終了フレームの指定は不要です。

```bash
boxer/bin/python run_boxer.py \
  --input sample_data/<記録名> \
  --neon_geometry output/tag_alignment/geometry_segment_000.npz \
  --start_n 0 --skip_n 1 \
  --gaze_targets --point_cloud off --track --force_cpu \
  --labels chair,table,monitor \
  --output_dir output/neon_aligned_000
```

姿勢がないフレームも2D・3D検出を行い、独立したカメラ基準の結果を保存します。
`boxer_frames.csv` の `has_world_pose` が世界座標との対応の有無を表し、混在する
出力の `coordinate_frame` は `mixed_world_and_per_frame` です。
視線・注視候補の座標系もフレームごとに切り替わります。
`boxer_3dbbs_world.csv` には世界座標の検出だけを保存し、融合にはこちらを使います。
座標合わせされていない複数のSLAMマップは別の出力先で処理します。
別のVIO/SLAMマップの点群を混ぜる場合、その点群も同じ実環境座標へ整列する
必要があります。この変換コマンドは点群や3Dモデルを生成しません。

## 5. Viewerで視線・注視位置を3D表示

`view_tracker.py` は同じ出力先の `boxer_gaze.csv` と
`boxer_gaze_targets.csv` を自動で読み込みます。右の3D画面に
**オレンジの視線**と**マゼンタの推定注視点**を表示し、RGBにも対応する位置を重ねます。
Gaze欄で表示のON/OFF、交点がない場合の視線の表示長を変更できます。
対象名・推定距離・欠測理由もGaze欄に表示します。

先ほどの点群・軌跡なしの出力を再生する場合：

```bash
boxer/bin/python view_tracker.py \
  --input sample_data/2026-10-10_09-39-40-d4fe821f-20261015T025507Z-1-001/2026-10-10_09-39-40-d4fe821f \
  --output_dir output/neon_gaze --point_cloud off --autoplay
```

姿勢がないフレームでは「No camera pose」と表示されます。画像パネルには
そのフレームの2D・3D検出と視線・注視候補を表示します。右の世界座標画面は
それまでのモデルを保持し、現在のカメラ・視線・注視点は表示しません。
軌跡が全くない入力では、世界座標のモデルは作成しません。

SLAM3またはTag Alignerの姿勢を使って推論した出力では、推論時のNPZパスを
自動で読み込みます。NPZを移動した場合や旧出力には、同じNPZを
`--neon_geometry` で指定してください。

```bash
boxer/bin/python view_tracker.py \
  --input sample_data/<記録名> \
  --output_dir output/neon_aligned_000 \
  --point_cloud off --autoplay
```

`--output_dir` は記録名サブディレクトリの**親**です。
`boxer_3dbbs.csv`、`boxer_input.json`、`boxer_frames.csv` を同じ場所に保管してください。
元のフレームID・int64 ns時刻で映像、姿勢、視線を照合します。
検出0件のフレームも再生し、時刻や座標が不整合なCSVはエラーにします。
Viewerの `--start_n` / `--skip_n` / `--max_n` はこの処理済みフレーム列に適用します。

視線CSVだけなら線のみ表示します。交点のない線の先端は注視点ではありません。
欠測時は該当する線・点を消し、前フレームの位置を残しません。
CSVがない既存の出力では、視線を表示せず通常の物体表示を行います。
注視点は保存済みの推定値なので、Viewerの物体信頼度フィルタや
オンラインTrackerの設定を変えても再計算しません。
軌跡があるモードのTrackerは、保存済み追跡履歴そのものではなく
`boxer_3dbbs.csv` から再計算します。

点群の `auto/on/off` はViewerでも利用できます。`on` はNPZ内の
登録済み `points_world` が必須です。視線・注視点の表示自体には点群は不要です。
画面は3Dボックス・軌跡・線・点の表示であり、部屋の面メッシュは生成しません。
対話操作にはOpenGL対応のGUIセッションが必要です。

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
視線CSV単体には注視深度や対象名を含めません。
`--gaze_targets` を指定した場合の交点・対象候補は別のCSVに保存します。
近距離の視差、視線・姿勢の誤差、フレーム内の時間差が残ります。
`world_ray` は座標変換ができたという意味であり、絶対位置精度を保証しません。
Tag Aligner姿勢の有無と由来は `boxer_input.json` の `alignment` で確認できます。
点群なしのBoxerによる物体の深度・サイズもモデルの推定です。

`--skip_viz` は画像出力だけを、`--no_csv` はCSV/メタデータ出力を無効にします。
`--cache3d` はフレーム読込を省略するため `--neon_gaze` と併用できません。
JPEG/MP4表示に加え、上記の `view_tracker.py` で対話的に3D表示できます。

## 検証

```bash
boxer/bin/python -m pytest -q tests/test_gaze_viewer.py tests/test_neon_viewer.py \
  tests/test_gaze_target.py tests/test_neon_gaze.py tests/test_neon_loader.py \
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

同じフレーム10～12に `laptop,cell phone` も含む5種類のラベルを指定した
点群なしの実行では、3フレームとも交点を得ました。
対象候補は `cell phone`, `cell phone`, `laptop`、推定距離はそれぞれ
約0.491 m、0.653 m、0.587 mでした。実測値との比較は未実施です。
この5ラベルの条件でも、注視推定を有効にした場合と無効にした場合で、
2D検出24件・3D検出23件・視線3行のCSVがバイト単位で一致しました。
全体テストは404件成功、任意依存関係のない6件をスキップしました。
