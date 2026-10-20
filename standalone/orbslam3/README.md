# Neonを独立したORB-SLAM3で処理する

Boxerの推論環境・SLAMビルド・途中終了制御から分離した実行系です。
生成物は `output/orbslam3_standalone/` に置きます。既存の解析結果は変更しません。
Python環境は専用venvでNumPyだけを使い、PyTorchやBoxerをimportしません。
ORB-SLAM3はCPU実装です。後段のBoxer推論はCUDA必須で実行してください。

## ビルド

既存の `output/neon_vio_deps/ORB_SLAM3` のGitオブジェクトから、公式コミット
`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` を別ディレクトリへ展開します。
既存ワーキングツリーの変更は取り込みません。Pangolin等のローカルprefixもコピーします。
追跡・初期化・最適化・ループ閉じ込み等の主要8ファイルは、公式コミットとバイト単位で照合します。
追加するのは `slam/export_neon.patch` のSystem状態取得・終了後の出力APIだけです。
C++17用の `LoopClosing.cc` 修正は適用せず、本体は依存ライブラリが要求する
C++14互換設定、入力runnerはC++17でビルドします。最適化は `-O3 -march=native`、
Eigenの配置条件は全対象で統一します。システムのOpenCV/Eigen/GCCは共有します。

```bash
bash standalone/orbslam3/build.sh
```

neon_Htouでは無変更の公式処理が約2,300フレームでクラッシュしました。
このデータを継続処理する場合は、後述する最小限の安全ガードを明示的に有効にします。

```bash
NEON_ORPHAN_GUARD=ON bash standalone/orbslam3/build.sh
```

ガードは `KeyFrame::SetBadFlag()` で、親が未設定のキーフレームの削除を見送るものです。
推定式・特徴点数・採用閾値・IMU初期化は変更しません。公式ソースは変更せず、
ビルド内の `KeyFrame_guard.cc` と `LocalMapping_guard.cc` に判定と診断ログを追加します。
呼び出し側でも、IMUの時間方向リンクや積分値を変更する前に削除を見送ります。
ガードOFFが公式処理との比較用、ONがクラッシュ回避の検証用です。
有効・無効は `execution.json` と `pipeline.json` に記録します。
同じ親なし・子ありのグラフを作る実C++テストは、次で実行できます。

```bash
LD_LIBRARY_PATH="$PWD/output/orbslam3_standalone/prefix/lib" \
  ctest --test-dir output/orbslam3_standalone/build --output-on-failure
```

初回はNumPyの取得が必要です。オフラインでは `NEON_NUMPY_WHEEL=/path/to/numpy.whl`
でローカルwheelを指定できます。PythonやC++依存を環境外へインストールしません。
元の依存チェックアウトとprefixがない環境では、先に既存の依存セットアップが必要です。

## 軌跡取得

Neon独自形式からのデコード・単位変換・歪み補正は、既存の前処理結果を入力契約として
再利用します。ORB-SLAM3がNeon録画を直接読み込めるという意味ではありません。
同じ画像・IMUで組み込み方だけを比較でき、画像は再圧縮しません。

```bash
output/orbslam3_standalone/.venv/bin/python standalone/orbslam3/run.py \
  --prepared output/neon_Htou_run_01/prepared \
  --output output/neon_Htou_standalone_04
```

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `--prepared` | 必須 | images/IMU/manifest/camera.yamlを持つ既存の前処理結果 |
| `--output` | 必須 | 新しい空の出力先 |
| `--start_n` | 0 | 元録画の開始フレームID |
| `--max_n` | 残り全部 | 元録画の選択フレーム数。先頭のIMU不足フレームも分母に含む |
| `--pace` | 1 | 1は記録時刻間隔、0は待機なし。0は地図作成スレッドの遅れに注意 |
| `--lost-keyframes` | on | onは公式既定値の喪失中KF追加を有効化。preserveは入力設定を維持、offは無効化 |
| `--min-frames` | 2 | 保存する有効な連続区間の最小長 |
| `--convert-only` | false | Boxer推論前の実行済み出力からNPZとpipeline.jsonだけを再生成 |
| `--diagnostics` | false | 別ビルドで初期化分岐・地図イベント・入力タイミングを記録する。閾値は変更しない |

### 初期化失敗と地図のリセットを診断する

```bash
NEON_ORPHAN_GUARD=ON NEON_DIAGNOSTICS=ON bash standalone/orbslam3/build.sh

output/orbslam3_standalone/.venv/bin/python standalone/orbslam3/run.py \
  --prepared output/neon_Htou_run_01/prepared \
  --output output/neon_Htou_init_diagnostics \
  --start_n 25834 --max_n 900 --pace 1 --lost-keyframes on --diagnostics
```

診断版は `build_diagnostics/` に作成し、通常版の `build/` を置き換えません。
公式ソースも変更せず、生成したコピーへ観測処理を加えます。
固定コミットの挿入位置が変わった場合はビルドを中止します。
`execution.json` に実際の実行バイナリとライブラリのSHA256を保存します。

- `slam/init_diagnostics.csv`：全入力画像に1行。初期化の基準画像設定、
  基準/現在画像の特徴点不足、対応点不足、IMU単眼の時間切れ、2画像復元不成立、
  初期BA後の検査不成立、視覚初期化成功、通常追跡を区別します。
  特徴点・対応点は本体の値です。`reference_age_s` は現在画像との差、
  `timeout_age_s` は公式の時間切れ判定が使う前画像との差です。
  特徴点不足と時間切れが同時成立した場合は特徴点不足を主理由とします。
  幾何学的失敗では選択モデルF/H、RANSAC内点、良好点、三角測量点、視差、
  姿勢の曖昧さを保存します。未評価の値は空欄で、ゼロに置換しません。
- `slam/map_events.jsonl`：要求の重複を抑えた`RESET_REQUEST`と、
  地図を実際に消去した後の`RESET_APPLIED`を区別します。
  `map_epoch`は同じ地図IDでも実際のリセットごとに増加します。
  地図作成/切替/統合、IMU初期化・最適化・尺度推定・バイアス、慣性BA段階、
  局所BAの開始/終了、各画像での状態と未処理KF数も記録します。
  `source_frame_id`は対象フレーム/KFの撮影時刻から特定し、不明ならnullです。
  非同期イベントで処理対象KFを現在画像として誤記しないよう、
  現在投入中の画像は別の`input_source_frame_id`に保存します。
  リセット理由の`Tracking_reset_site_*`は生成Trackingソース内の呼出し位置です。
  BAの終了イベントは関数から戻ったことを示し、収束や位置精度の保証ではありません。
- `slam/timing.csv`：画像投入間の実時間、記録時刻から計算した投入予定に対する遅れ、
  追跡処理時間。`pace`を変えても画像・IMUの計測時刻は変えません。

観測用ログとロックも並列処理のタイミングに影響し得ます。診断あり/なしの単発の差を
精度改善と見なさず、比較対象の診断設定を揃えて繰り返します。
難区間への進入試験は前区間を取得率の分母に含めず、進入直前の初期化状態・地図ID・
epochと、難区間中のリセットを確認します。進入前に地図を失っていた場合、
「初期化済みでも通過できない」と判定してはいけません。

まず `--max_n 900 --lost-keyframes preserve` で短区間を比較し、その後に設定を一つずつ
変更してください。別ソース・ビルド条件の比較結果を、一つの変更だけの効果とは断定できません。

このrunnerは一つのSystemを最後まで使います。追跡喪失・リセット時にも外から終了・再起動
しません。地図の保持・切替・ループ閉じ込み・統合は元のORB-SLAM3に任せます。
ただし元のORB-SLAM3自身がactive mapをリセットして履歴を破棄する場合はあります。
再訪や対応がなければ地図は統合されず、一つの世界になる保証はありません。

## Boxerへ渡す

```bash
.venv/bin/python run_neon_pipeline.py \
  --input sample_data/neon_Htou \
  --output output/neon_Htou_standalone_04 \
  --boxer-only --require_gpu --skip_viz --gaze_targets
```

独立処理が生成した `pipeline.json` を既存の `--boxer-only` で読み込みます。
これは全候補地図を処理する汎用的な接続例です。下記のneon_Htou全録画試験では
一部地図の発散を確認したため、その試験の全候補を無条件に3D構築へ使わないでください。
軌跡推定を繰り返さず、選択した全フレームの検出と一つのViewerタイムラインを作ります。
軌跡のない画像はカメラローカルの3D検出に使い、世界座標への配置には使いません。

出力 `geometry/geometry_segment_*.npz` は、元のint64ナノ秒timestamp、
`T_world_camera`（OpenCV Scene Camera→重力整合したORB世界、メートル）、
任意の同世界の `points_world` を保持します。IMUボディ姿勢をカメラ姿勢として渡しません。
最終最適化後の姿勢のうち、撮影時の追跡もOK・IMU初期化済みだったものだけを採用し、
補間しません。点群は3回以上観測された有効点です。区間ごとに原点を移動しません。

現在のBoxer Viewerは連続区間単位で世界を切り替えます。同じ最終map_idの区間であっても
自動的に累積融合する機能は追加していません。独立したmap_id同士の位置合わせも行いません。
そのため「一つのSystemで処理した」と「一つの連続地図を取得できた」は区別してください。

同じ最終地図を一つのViewer世界として使いたい場合は、追加出力の
`geometry/map_*.npz` を直接Boxerへ渡してください。これは途切れた区間も含めて同じ
map_idの有効な姿勢をまとめたもので、欠落区間を補間しません。
`pipeline.json` の `standalone.map_outputs` に地図別のフレーム数とパスがあります。
次はmap_000を選ぶ例です。実際には出力された地図を確認して選びます。

```bash
.venv/bin/python run_boxer.py \
  --input sample_data/neon_Htou \
  --neon_geometry output/neon_Htou_standalone_03/geometry/map_000.npz \
  --neon_imu_config output/neon_Htou_standalone_03/prepared/manifest.json \
  --output_dir output/neon_Htou_map000_boxer \
  --track --require_gpu --skip_viz --gaze_targets

.venv/bin/python view_tracker.py \
  --input sample_data/neon_Htou \
  --output_dir output/neon_Htou_map000_boxer --require_gpu --autoplay
```

元録画全体を検出対象にし、選んだ地図に姿勢があるフレームだけを世界座標の構築に使います。
別のmap_idを同じ世界に重ねません。区間別のpipeline出力と地図別のNPZでは、
前者が最小連続長未満を除外するため、採用フレーム数が異なる場合があります。

`execution.json` に実行コマンド・バイナリhash・校正設定hash、`slam.log` にORBログ、
`slam/states.csv` に全フレームの状態、`progress.log` に進捗、`pipeline.json` に採用率を保存します。
取得率や再投影の整合性は、独立した基準軌跡による絶対精度とは異なります。

## 実データでの接続検証

neon_Htouの元フレーム9988–10317の330枚を、同じ前処理画像・校正・IMU・
PACE=1・喪失中KF追加なし・最小長2で各1回実行しました。

| 方式 | 有効フレーム | 取得率 | 最初の実行時IMU初期化済み追跡成功フレーム |
| --- | ---: | ---: | ---: |
| 現在のBoxer組み込み方式 | 292 | 88.48% | 10026 |
| 独立方式 | 235 | 71.21% | 10083 |

この区間では両方式とも初期化後は最後まで追跡でき、差は初期化成立時点の違いでした。
各1回・ビルド条件も異なるため、差を途中終了制御だけの影響と断定できません。
独立化による取得率向上や絶対精度向上は、この試験では確認していません。
実行条件と集計は `output/orbslam3_standalone/comparison_330.json` にあります。

独立方式の出力を既存の `--boxer-only --require_gpu --skip_viz --gaze_targets`
へ渡し、235枚の世界座標3D推論と95枚のカメラローカル推論、全330枚のViewer用出力を
RTX 3070 Laptop GPUで完了しました。実際のViewerもNVIDIA OpenGLで全330枚と
逆方向シークを検証しました。結果は `output/orbslam3_standalone/validation_330/`、
描画検証は同階層の `viewer_result.json` です。

地図単位NPZについても、検証用コピーから姿勢を1枚だけ取り除いたケースを用意し、
234枚の世界姿勢と96枚のローカル推論を含む全330枚をGPUで処理しました。
Viewerは欠落を挟んでも世界を分割せず、全330枚と逆方向シークをNVIDIA OpenGLで
描画できました。`map_gap_gpu/` と `map_viewer_result.json` がその検証記録です。
姿勢を取り除いたNPZは検証専用で、実測の解析出力には使いません。

無変更版の全録画試験は `output/neon_Htou_standalone_01/` に保存しています。
2,299枚処理後にSIGSEGVで停止し、最終軌跡は未出力です。デバッガ付きの再現でも
2,310枚付近で `LocalMapping::KeyFrameCulling → KeyFrame::SetBadFlag →
ChangeParent → AddChild → pthread_mutex_lock(0x1128)` を確認しました。
逆アセンブルでも、親のnullポインタにmutexのオフセット0x1128を足していることを確認しました。
スタックは `output/orbslam3_standalone/debug_2400/gdb_host.log` にあります。
親が未設定になるまでの全経緯は未確定です。喪失中KF追加設定との因果も未確定です。
異常終了は `failure.json` に記録し、失敗した出力のNPZ変換・Boxerへの接続は拒否します。

## 全録画試験と品質の限界

安全ガードON、喪失中KF追加ON（公式既定値）、PACE=1で全31,404枚のVIO入力を
終了まで処理しました。所要時間は約3,141秒です。元録画はIMU不足の先頭1枚を含め31,405枚です。
結果は `output/neon_Htou_standalone_03/` に保存しています。既存run_01/run_02は変更していません。

連続長2以上の判定を通った候補は5,053枚（16.09%）、84区間でした。
地図単位NPZには孤立した有効姿勢も含め5,105枚（16.26%）があり、座標系は4個です。
run_02の2,940枚（9.36%）より候補数は増えていますが、**精度向上を確認した結果ではありません**。
ビルド設定・喪失中KF追加設定・セッション制御も異なり、各条件1回の試験です。

| 最終地図 | 候補姿勢 | 連続フレーム間速度の中央値 / 最大値 | 判定 |
| --- | ---: | ---: | --- |
| map_000 | 587 | 0.36 / 1.92 m/s | 基本的な移動量に破綻なし。絶対精度は未検証 |
| map_001 | 1,358 | 1.08 / 143.69 m/s | 室内歩行として不成立。点群もkm規模で発散 |
| map_002 | 1,030 | 0.56 / 10.69 m/s | 急な位置飛びがあり要確認 |
| map_003 | 2,130 | 441.43 / 1,538.02 m/s | 室内歩行として不成立。点群も約100km規模で発散 |

異常はカメラ軌跡だけでなく、SLAM内部からそのまま保存した地図点群にも見られます。
IMU初期化済みフラグだけでは、実寸の正しさや推定の安定性を保証できません。
発散の原因について、同期・校正・初期化・喪失中KF追加・安全ガードの影響を
完全に切り分けられていません。全録画の実用的な単一3Dモデルは未達成です。
異常なフレームを閾値で隠して取得率を作り直す処理はしていません。

`pipeline.json` の `standalone.map_outputs` に各地図の連続速度とカメラ範囲を保存します。
詳細集計は `quality_audit.json` と `output/orbslam3_standalone/full_result.json` にあります。
この試験からBoxerへ実際に接続する検証には、比較的整合したmap_000を選びます。
上記の直接入力例はその地図を指定しています。これは地図全体の絶対精度保証ではありません。

最終map_000の元フレーム138–167の30枚でも、RTX 3070 Laptop GPUによるBoxer推論と、
NVIDIA OpenGLによる全30枚・逆方向シークの描画を完了しました。
出力は `output/orbslam3_standalone/final_map_gpu/`、描画記録は
`output/orbslam3_standalone/final_viewer_result.json` です。
全録画のBoxer再推論は、不良地図を含むため実行していません。

Pythonの全体検証は `PYTHON=.venv/bin/python bash scripts/ci.sh` で529件成功、12件スキップでした。
スキップにはCIからCUDAを隠しているテストが含まれ、上記の実GPU検証は別に行っています。
ガードの実C++回帰テストも成功しています。

### Fixed-input IMU initialization audit

The optional diagnostic build can copy the numerical graph inside the first
`Optimizer::InertialOptimization` overload, after its vertices/edges have been
assembled. This does not sort keyframes or change estimator parameters.

```bash
NEON_ORPHAN_GUARD=ON NEON_DIAGNOSTICS=ON bash standalone/orbslam3/build.sh
output/orbslam3_standalone/.venv/bin/python standalone/orbslam3/run.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/imu_capture_example --start_n 9988 --max_n 900 \
  --pace 1 --diagnostics --imu-snapshots
.venv/bin/python standalone/orbslam3/compare_imu_initialization.py \
  output/imu_capture_example/slam/imu_snapshot_1.json \
  --repeats 5 --output output/imu_replay_example
```

`--imu-snapshots` defaults off and requires `--diagnostics`. Capture adds I/O
and can influence SLAM thread scheduling; it cannot guarantee a byte-identical
SLAM run. Every capture therefore has a native solver result, and replay must
match that result before comparisons are interpreted. Captures contain numeric
copies, not serialized keyframe pointers. Matrices are row-major arrays of
17-digit decimal strings; cast them to numbers when reading JSON. Pose/velocity,
shared biases, gravity, scale, vertex/edge insertion order, fixed flags,
preintegration covariance and Jacobians (including the edge's cached copies),
priors and solver settings are preserved. The original prepared camera YAML and
manifest accompany the snapshots, including noise/extrinsics/time-offset sources.
Preintegration data are observed from the object used by the actual edge; the
native/replay check is necessary to detect a mutation or incomplete capture.

`*.after_scale.json` and `*.after_full_ba.json` record actual camera centers
following map scale application and following full-BA map correction. Compare
shared source frames: a scale multiplier alone is not an absolute accuracy test.
The replay executable loads no images and creates no SLAM threads. It uses the
same compiled g2o and IMU edge implementations as the capture. It reproduces the
200-iteration Levenberg solver, including its conditional initial lambda; the
upstream `bGauss` argument is unused in this overload. Fixed pose camera
extrinsics are recorded for interpretation; only body pose is consumed by the
IMU-only equations. Later full BA and the complete trajectory are not replayed.

### Independent correspondence benchmark

```bash
.venv/bin/python standalone/orbslam3/benchmark_neon_correspondences.py \
  --audit output/neon_Htou_branch_audit_01 \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/correspondence_example
```

This reuses the exact 370 fresh-reference failed pairs and adds 100 distinct
control references from a previously successful tracking window. For each
reference, offsets 1/2/3/5 are separate conditions. The unchanged native ORB
initialization matcher must reproduce all 370 logged counts exactly. Features
are extracted with the upstream initialization budget (5 times nFeatures).
Multiscale ORB uses the same descriptors, removes spatial duplicates while
preferring fine scales (pairwise radius 3 times the larger octave sigma), and applies Hamming <=50, ratio <0.9, mutual matching.
An additional `level0_orb_control` uses that same comparison matcher with only
octave 0, so its contrast with multiscale isolates octave eligibility. Comparing
native ORB with the multiscale method alone also changes window, assignment and
orientation filtering; that difference cannot be attributed to octaves alone.
KLT uses up to 2,000 goodFeaturesToTrack corners and a 1px forward/backward gate.

All methods receive the same conservative 1.5px essential-matrix RANSAC test.
Triangulation requires positive depth, >=1 degree parallax and <=3 octave-sigma
pixel reprojection error. The summary gate also requires 100 essential inliers,
50 triangulated points, coverage of six cells in a 6x4 image grid, and 50
triangulated correspondences continuing into a third frame. This is an explicitly
chosen diagnostic proxy, not the ORB-SLAM3 initialization acceptance test or
proof of metric trajectory accuracy. Essential-only reconstruction can remain
ambiguous in planar or rotating scenes. Three-frame continuation is a matching
check, not three-view bundle adjustment. Counts, geometry, distribution and
continuation are also saved separately so a failed combined gate is interpretable.
Native OpenCV and Python OpenCV versions may differ; the native matcher remains
unchanged and the common geometry uses one Python OpenCV version for all methods.
These classical CV routines run on CPU; this benchmark performs no neural
inference. Any future learned matching experiment must require CUDA explicitly.

The diagnostic executable `audit_inertial_scale_jacobian` checks the upstream
scale Jacobian against central finite differences of the actual `VertexScale`
update. In this pinned revision, `VertexScale` updates `s` by `s*exp(delta)`,
but the IMU edge's scale column omits the chain-rule factor `s`. The executable
checks scales 1, 2, and 10 and reports both the upstream discrepancy and the
error after multiplying its column by `s`.

For a separately labelled **replay-only experiment**, pass
`--correct-scale-jacobian` to `compare_imu_initialization.py` (or as the last
argument of the native replay executable). That experiment overrides only the
scale column in the reconstructed graph. It does not change the live SLAM
estimator. Native objective values, repeated replay objective values and solver
estimates are saved; lower residual does not establish metric trajectory accuracy.
The observed after-scale/full-BA checkpoints belong to the original native run,
not to this hypothetical replay condition.

If classical matching remains insufficient, the optional learned benchmark uses
[official SuperPoint + LightGlue](https://github.com/cvg/LightGlue), pinned to
`eb42fee2d71449efb0aa5c10549752b5d75384d8`. Obtain that source in
`output/neon_lightglue_runtime/source` and place Kornia dependencies under
`output/neon_lightglue_runtime/deps`; no application environment edit is needed.

```bash
.venv/bin/python standalone/orbslam3/benchmark_neon_lightglue.py \
  --classical output/neon_Htou_correspondence_audit_01 \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/neon_Htou_lightglue_audit_01
```

CUDA is mandatory. Feature extraction and matching use Float32, with mixed
precision, FlashAttention's internal half conversion, and TF32 disabled.
Original 800x600 coordinates, exact pair/control identities, thresholds and
three-frame continuity tests match the classical benchmark. Official weights
are downloaded inside the experiment runtime, and their hashes plus the GPU,
package versions and feature counts are recorded. Learned correspondences are
an independent comparison; they are not passed into ORB-SLAM3's ORB descriptors.

For the recorded experiment, the isolated dependencies were installed with:

```bash
git clone https://github.com/cvg/LightGlue.git output/neon_lightglue_runtime/source
git -C output/neon_lightglue_runtime/source checkout eb42fee2d71449efb0aa5c10549752b5d75384d8
uv --cache-dir output/neon_lightglue_runtime/uv-cache pip install \
  --python .venv/bin/python --target output/neon_lightglue_runtime/deps \
  --no-deps kornia==0.8.3 kornia-rs==0.2.0
```

Measured on 2026-10-19, `output/neon_Htou_optimizer_audit_02/report.txt`
contains the full local report. Eleven sequential 900-frame captures obtained
10 runs without a reset after the first IMU initialization and one with an early
reset. The early-reset run later recovered (87.33% candidate coverage), so this
classification describes its first initializer, not failure of the entire run.
Both saved inputs reproduced their native estimates exactly in five fresh replay
processes. Their nine shared IMU constraints were identical, but their initial
visual poses, velocities and gravity estimates differed. Reordering the stable
input's graph in 13 separate conditions changed estimates by at most 4.09e-14.
The exact visual update that first created the different map geometry remains
unidentified.

The fixed hard-reference correspondence results were:

| Method | +1 frame | +2 frames | +3 frames | +5 frames |
| --- | ---: | ---: | ---: | ---: |
| Native ORB | 0/370 | 0/370 | 0/370 | 0/370 |
| Level-0 comparison control | 0/370 | 0/370 | 0/370 | 0/370 |
| Multiscale ORB | 0/370 | 0/370 | 0/370 | 0/370 |
| KLT | 13/370 | 8/370 | 5/370 | 3/370 |
| SuperPoint + LightGlue (CUDA Float32) | 154/370 | 194/370 | 181/370 | 154/370 |

Learned matching passed the common gate at at least one offset for 287/370
references, including 133 that failed at +1 but succeeded later. These are
independent geometry/continuation results, not SLAM trajectory coverage or
absolute pose accuracy. Calibration recordings and learned-feature SLAM/map
integration remain outstanding. The full recording was not rerun.

The replay-only Jacobian correction reduced stable-input chi2 from 29942.19 to
29929.40, but increased early-reset-input chi2 from 32195.44 to 32248.01.
It is consequently retained as a separately labelled experiment; the live
upstream estimator is unchanged. A mathematically consistent derivative alone
has not been shown to solve the observed initialization instability.

### Independent multi-view visual reconstruction

`neon_sfm.py` evaluates actual shared-point reconstruction, independently of
Boxer's metric pipeline. It requires the above LightGlue runtime and the earlier
`neon_Htou_lightglue_audit_01/results.csv` for the **predeclared** case and seed
selection rules. The optional COLMAP wheel lives outside the application venv:

```bash
uv --cache-dir output/neon_sfm_runtime/uv-cache pip install \
  --python .venv/bin/python --target output/neon_sfm_runtime/deps \
  --no-deps pycolmap-cuda12==4.2.1
.venv/bin/python standalone/orbslam3/neon_sfm.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/neon_Htou_learned_sfm_01
```

The default experiment covers **every image from 25834 through 26733**. Stages
`extract`, `match`, `triplets`, `database`, and `reconstruct` can be run separately;
`--resume` requires identical source IDs, original nanosecond timestamps and
calibration identity. Learned inference requires CUDA and Float32. The pinned
CUDA COLMAP wheel's Ceres build lacks a CUDA BA solver; BA and classical geometry
explicitly use CPU, rather than silently falling back from a GPU request.

Correspondences use offsets 1/2/3/5. Three fixed groups each contain the first five
distinct references satisfying the previous adjacent-pass, delayed-rescue or
failed-gate condition. Each case triangulates two views, estimates camera 3 by
PnP, runs joint 3-view BA, estimates camera 4 against the **same** updated 3D
points, then runs joint 4-view BA. Failures remain in `triplets/results.json`.
Essential-matrix cheirality alternatives, homography support and point planarity
are diagnostic fields, not proof of unique geometry. All BA intrinsics stay fixed.

The default pipeline then reconstructs 40 contiguous images and all 900 images
with COLMAP. It does not concatenate independently normalized relative motions.
For seed sensitivity, compare **identical single-map settings**:

```bash
.venv/bin/python standalone/orbslam3/neon_sfm.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/neon_Htou_learned_sfm_01 --resume \
  --stage reconstruct --single-map --seed-policy earliest
```

Repeat that comparison with `auto` and `most_points`. `earliest` selects the first
previously passing pair within the first 60 references; `most_points` maximizes
its triangulated count in the same candidate set, with source-ID tie breaks.
Forced seeds require `--single-map`: otherwise COLMAP can keep trying to reuse
that initial pair when starting later models. An interrupted preflight exercising
that behavior was retained and excluded, not counted as a completed experiment.

Each model has a quality report and an explicitly **nonmetric** pose export.
COLMAP camera-from-world poses are inverted. `visual_trajectory.npz` stores
original `int64` source IDs/timestamps, 3x4 camera-to-world matrices and
`metric=False`; it deliberately does not emit a Boxer `pipeline.json` or a metric
geometry NPZ. Feature coordinates **and** principal point both move by +0.5px
exactly once, preserving the OpenCV normalized rays. Independent maps are not
aligned; the sum of model sizes may double-count overlapping frames. Reports
therefore separate unique registered frames, largest map, contiguous spans,
holes, point track lengths, observation reprojection errors and negative depth.
This offline method uses future images and is not an online SLAM accuracy test.

### Visual graph and descriptor replay

```bash
output/orbslam3_standalone/.venv/bin/python standalone/orbslam3/run.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/visual_capture_example --start_n 9988 --max_n 65 \
  --pace 1 --diagnostics --imu-snapshots --visual-snapshots \
  --visual-max-source 10025
LD_LIBRARY_PATH="$PWD/output/orbslam3_standalone/prefix/lib" \
  output/orbslam3_standalone/build_diagnostics/replay_visual_ba \
  SNAPSHOT.json REPLAY.json
.venv/bin/python standalone/orbslam3/compare_visual_maps.py \
  RUN_A/slam RUN_B/slam --output COMPARISON.json
```

Visual snapshots default off. They cover initial two-view geometry, initial BA
and normalization, point culling/creation/fusion, KF culling, tracking pose
optimization, actual visual BA graph input/result and separate map writeback.
Collection stops after IMU initialization, a reset epoch, or the explicit source
limit. Synchronous observation changes scheduling and allocation; capture runs
are diagnostic evidence, not an unbiased latency/coverage benchmark.

Map snapshots also preserve representative ORB descriptors, normals, distance
limits, ordered covisibility neighbors and feature-to-point slots. Points are
matched across runs through shared `(source frame, feature index)` observations,
never by native MapPoint ID alone. Sim(3) comparison uses shared points and camera
centres and reports relative rotation separately. An existing single-pose scale
gauge is recorded, not assumed to explain every deformation.

Visual replay restores exact stored quaternion coefficients, graph insertion
order, fixed/marginalized variables, pinhole intrinsics, information matrices,
Huber kernels and actual iteration count. Cached errors are observed without
recomputing them in live BA: recomputing could change native outlier removal.
Detached analysis separately recomputes the objective. Abort-interrupted graphs
may depend on the inner stopping instant; exact replay is checked on completed
graphs. The optional replay-only `--anchor-scale` adds one scalar log-baseline
residual, not a fixed second pose. It has a synthetic gauge regression test; it
was **not** applied as a real-data fix because the first observed inputs already
contained different descriptor choices/observations.

`replay_point_descriptors.py SNAPSHOT --features FEATURES.bin ... --output RESULT`
reconstructs upstream Hamming medoid selection from saved native keyframe
features and the snapshot's original observation order. A separate canonical
source/feature order is only a diagnostic control. It does not modify live SLAM.

### Scale Jacobian numerical reference and convergence

```bash
LD_LIBRARY_PATH="$PWD/output/orbslam3_standalone/prefix/lib" \
  output/orbslam3_standalone/build_diagnostics/replay_imu_initialization \
  SNAPSHOT.json RESULT.json --numeric-scale-jacobian --trace
```

Compare this with `--correct-scale-jacobian` and neither flag. `--warm-start
NATIVE_RESULT.json` initializes free variables at the native final solution;
fixed poses and graph constraints stay unchanged. Include a native-Jacobian
warm-start control because restarting also resets damping. `--trace` records
outer-iteration objective values, actual accepted/attempted step norms, inner
rejection counts, damping, numerical tangent-space gradient norm and termination
status. Gradient norms mix parameter units and are not a metric accuracy score.
The numerical reference replaces **only** the scale column with central
differences of `s*exp(delta)` at epsilon 1e-6. All conditions remain replay-only.

### Recorded 2026-10-20 experiment

- Four-view reconstruction passed 2/5 adjacent cases, 3/5 delayed-rescue cases,
  and 1/5 previously failed-gate cases. All 40 contiguous images registered in
  one map. The 900-image automatic multi-map run registered **883/900 unique
  images (98.11%)**, across **11 maps**; its largest single map was **266/900
  (29.56%)**, continuously spanning **26176–26441**. Mapping took 294.2s, excluding
  feature extraction/matching. These are arbitrary-scale registration figures,
  not metric coverage or independent accuracy measurements.
- Identical single-map seed comparisons registered **3, 204, 204** images for
  automatic, earliest and most-triangulated selection. The automatic 3-image,
  3-point model is an unusable small reconstruction despite its low residual.
  Seed sensitivity is substantial. The automatic multi-map total must not be
  presented as a 98% single-world result.
- Visual capture cohort 1 recorded an early reset in trial 5. Initial geometry,
  initial BA and normalization were identical; first recorded observation
  differences occurred in neighbor fusion at source 10004. Local BA input had
  1111 versus 1092 observations. Cohort 2 added descriptor/neighbor observations:
  all 20 short prefixes avoided the early reset, illustrating observer effects.
  In trials 1 and 3 its **first saved state difference** was descriptor/normal
  recomputation during `ProcessNewKeyFrame` at **10004**: 48/283 descriptors
  differed while point geometry and observation sets still matched. Native
  descriptor replay matched **283/283 in both runs**; all 48 differing choices
  were tied medoid minima resolved by native pointer order. Canonical source
  ordering made all 48 identical in the detached control. Subsequent fusion
  changed constraints, and first local BA changed shape (about 1.94% normalized
  point RMS after Sim(3)), not just uniform scale. This proves an input-variation
  mechanism; it does not prove that descriptor ties alone caused the original
  full-recording failure. No live matching/BA algorithm was changed.
- Initial and first local visual BA inputs each replayed five times exactly
  (all saved estimates: max difference **0**) for both selected runs. The
  corrected IMU scale derivative agreed with the numerical-column reference:
  scale differences below 1e-8 and objective differences below 1.6e-7. Actual
  native/corrected iteration counts were **9/8** for the stable input and **7/4**
  for the early-reset input, despite a requested limit of 200. The early case
  hit the upstream three-small-improvements rule; other conditions hit the
  ten-inner-trials limit. Warm-start comparisons and full traces were retained.
  Correct differentiation is verified; general convergence and metric accuracy
  remain unproven, so the production estimator is unchanged.

Local artifacts: `output/neon_Htou_learned_sfm_01/`,
`output/neon_Htou_visual_audit_01/`, `output/neon_Htou_visual_audit_02/`, and
`output/neon_Htou_jacobian_audit_03/`. Calibration recordings/independent metric
reference remain unavailable. Single-map continuity, holdout validation and
metric conversion are still required before connecting this route to Boxer.

Validation: `.venv/bin/python -m pytest -q tests` completed with **558 passed,
12 skipped** (55.45s). Native CTest completed **2/2 passed** with the runtime's
absolute `prefix/lib` on `LD_LIBRARY_PATH`. A 20-frame diagnostic smoke run with
both snapshot flags omitted completed with **zero visual/IMU snapshots**.
Generated datasets, feature/model weights and diagnostic outputs are not committed.

### Connecting existing maps and bounded boundary matching

`neon_map_connect.py` audits every pair of existing reconstructions using source
image IDs and feature-observation identities. It records common image spans,
camera-centre singular values, rotation spread, common 3D points, a training
Sim(3), held-out point residuals and camera pose consistency. The initial gates
require at least three shared images and 30 point correspondences, >=80% training
consensus within 1% of target point RMS radius, held-out 95th-percentile residual
<=2% of that radius, and camera rotation 95th percentile <=2 degrees. Rejected
pairs remain in the report. A common image alone is not sufficient to establish
scale, and geometric rank alone does not prove good conditioning.

```bash
.venv/bin/python -m standalone.orbslam3.neon_map_connect \
  --baseline output/neon_Htou_learned_sfm_01/mapping_auto_900 \
  --output output/neon_Htou_map_connect_01
.venv/bin/python -m standalone.orbslam3.neon_sfm_continue \
  --baseline output/neon_Htou_learned_sfm_01 \
  --output output/neon_Htou_boundary_sfm_01 --stage prepare
.venv/bin/python standalone/orbslam3/neon_sfm.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/neon_Htou_boundary_sfm_01 --stage match --resume
.venv/bin/python standalone/orbslam3/neon_sfm.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/neon_Htou_boundary_sfm_01 --stage database --resume
.venv/bin/python -m standalone.orbslam3.neon_sfm_continue \
  --baseline output/neon_Htou_learned_sfm_01 \
  --output output/neon_Htou_boundary_continuation_01 --stage continue \
  --database output/neon_Htou_boundary_sfm_01/database.db \
  --images output/neon_Htou_boundary_sfm_01/images
```

New experiments reuse the original Float32 feature files without extraction.
`pair_manifest.json` is the single input to both matching and DB import; its
SHA-256 is stored in completion markers. Changed manifests invalidate completed
DBs. A seed DB containing pairs outside the manifest is rejected. Image IDs and
original keypoint indices are preserved by copying the old DB into a new output.
Only missing pairs are matched on CUDA. Every pair missing its two-view
geometry undergoes verification, including interrupted imports whose raw matches
already exist. `pair_import_audit.json` lists raw/geometric counts for
every requested pair. Reconstructions use their own additional working DB copy:
COLMAP may update SQLite bookkeeping even when consuming an existing DB.

The bounded initial proposal straddles 26169→26176 and 26441→26447, with intervals
6/7/8/10 and up to ten surrounding frames on each side (18 new pairs). A subsequent
explicit 3×3 grid at 26410/26420/26430 versus 26450/26460/26470 is recorded as a
separate development experiment after the first grid found no geometric support
at the second boundary. Every continued run starts from the saved 266-image map,
refines both existing structure and new points with fixed intrinsics, exports
`metric=False`, and separates retained, removed and added observations. Keep a
no-extra-pairs continuation control to distinguish new matching from restarting
optimization. Counts and average residuals alone do not qualify a connection.

### Descriptor tie A/B experiment

The diagnostic build now generates an optional copy of `MapPoint.cc`; the pinned
upstream source remains untouched. `--descriptor-ties native` is the default.
`--descriptor-ties source` changes **only tied minimum medians** to use source
frame ID, camera stream index, and feature index. A standalone process contains
one recording, so recording identity is constant in this ordering. It does not
use keyframe creation IDs or pointer values as tie-break keys. The normal/depth
summation implementation, descriptor distance/median definition, feature budgets,
matching thresholds, IMU settings and BA settings remain unchanged.

Stable keys are resolved on demand without an auxiliary candidate allocation;
A retains the upstream candidate allocation and selection. The earlier preflight
that allocated a candidate-key vector even in A was excluded because it changed
the heap layout central to this investigation; attempted runs are preserved in
`output/neon_Htou_descriptor_ab_01/EXCLUDED.json`.

```bash
NEON_DIAGNOSTICS=ON NEON_ORPHAN_GUARD=ON bash standalone/orbslam3/build.sh
output/orbslam3_standalone/.venv/bin/python standalone/orbslam3/run.py \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/tie_example --start_n 9988 --max_n 65 --pace 1 \
  --diagnostics --descriptor-ties source --light-maps
.venv/bin/python -m standalone.orbslam3.descriptor_ab_report \
  output/neon_Htou_descriptor_ab_02
```

`--light-maps` saves compact point/descriptor/observation and camera-centre CSV
summaries at selected pre-IMU milestones. Heavy visual/IMU snapshots stay off.
The formal protocol fixes 20 runs per condition, alternates A/B, and uses the same
binary, orphan guard, lost-KF setting and minimal logging. Observation can still
influence asynchronous timing, so conclusions apply to this measured protocol.
Count resets **before** initial IMU initialization as well as afterwards. A late
initializer has less than twenty frames of follow-up in a 65-frame recording;
report this censoring rather than treating it as evidence of no early failure.
The optional policy is not promoted based solely on reproducibility; tracking
continuity and geometry must also remain useful. The scale-Jacobian experiment
remains independent and unchanged.


Run the complete, predeclared comparison in a **new** output directory:

```bash
.venv/bin/python -m standalone.orbslam3.descriptor_ab_experiment \
  --prepared output/neon_Htou_standalone_03/prepared \
  --output output/ties_new_comparison --trials 20
.venv/bin/python -m standalone.orbslam3.descriptor_ab_report \
  output/ties_new_comparison
```

### Measured outcome: connection and tie policy not accepted

For this 900-frame development interval, only three of 55 map pairs shared
registered images. None passed the common-observation consistency gates; COLMAP's
reprojection-based alignment also rejected all three. The largest 266-image map
had no shared registered images with the other maps. Existing overlap alone
therefore cannot extend it.

Boundary matching reused GPU Float32 SuperPoint features and added 18 pairs,
then a separate nine-pair grid. Continuing the largest map produced candidates
with 289 and 290 registered images, respectively. The latter increased the
collection's unique registration count from 883 to 886, but the longest continuous
interval stayed at 266. **These are rejected candidates, not usable improvements.**
Withholding the entire 26169→26176 match edge gave cross-projection median/p95
errors of 14.24/22.43 pixels; 42 of 58 directional tests lacked a supporting 3D
point, and none of the three pairs triangulated on both sides shared a landmark.
Keeping only calibrated new pair geometries did not extend the original 266.
The native structure-less registration fallback occurred in expanded candidates;
registration counts do not establish successful existing-point PnP or accuracy.
No merge was forced and no interpolation was counted as registration.

In the formal 20+20 alternating descriptor comparison, A (native) reset in 0/20
runs and retained a median of 27 metric-flagged candidate frames out of 65.
B (stable tied-minimum selection) reset in 20/20 and retained a median of three.
All B runs first initialized at source 10049, leaving fewer than 20 follow-up
frames; they cannot be counted as successful early survival. Within-condition
lightweight descriptor/fusion summaries were already repeatable in A, so this
protocol did not demonstrate a reproducibility benefit. **Keep native as default;
do not expand B to 900 frames.** Metric flags do not establish physical accuracy.

The original 883-image, 11-map result and exact original DB SHA-256 remain
preserved. A continuation control changed only SQLite bookkeeping in the original
DB before working-copy isolation was added; table contents were verified identical
and the exact original binary hash was restored. Subsequent runs use independent
DB copies and check source hashes. The active result remains 883 unique frames,
maximum single map 266, eleven independent coordinate systems, `metric=False`.
No full-recording or second hard-interval expansion was attempted after the
connection quality gate failed.

Detailed development evidence is in
`output/neon_Htou_map_connect_01/report.txt` and
`output/neon_Htou_descriptor_ab_02/analysis.json` (local, not committed).
Validation: **563 passed, 12 skipped** in the repository suite; native CTest
**2/2 passed**. The committed alternating-run driver also completed an actual
one-run-per-condition smoke comparison. Generated datasets, models and snapshots
are excluded from Git.

The later driver smoke test is excluded from the fixed 20+20 comparison: both
conditions completed but retained zero metric-flagged candidates, including A.
This also limits the repeatability claim to the formal measured batch; the
native system is not proven stable across separate launches or timing conditions.
