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
