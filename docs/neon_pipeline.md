# Neonの解析とRemote SSHでの結果確認

## COLMAPを選ぶ場合

従来のORB-SLAM3経路は既定のまま維持しています。
`--pose_backend colmap`ではSuperPoint＋LightGlue＋COLMAPを使用し、ORB用バイナリや語彙ファイルを必要としません。
現在は**非実寸プレビュー**として使用します。既存の復元結果を使う例です。
出力先には新しいディレクトリを指定してください。

```bash
python run_neon_pipeline.py \
  --input sample_data/neon_Htou \
  --output output/neon_colmap_good_02 \
  --prepared output/neon_Htou_standalone_03/prepared \
  --start_n 9988 --max_n 900 \
  --pose_backend colmap \
  --colmap_source output/neon_pose_comparison_02/Q-good_C0_01/mapping_auto_900 \
  --colmap_map_policy largest \
  --nonmetric_preview --point_cloud off \
  --require_gpu --skip_viz

python view_tracker.py \
  --input sample_data/neon_Htou \
  --output_dir output/neon_colmap_good_02/viewer \
  --require_gpu --autoplay
```

900枚すべてをGPUで物体検出・3D推論し、一つのViewerで再生します。
左・中央は各画像の検出結果、右はCOLMAPのカメラ軌跡・カメラ枠を表示します。
**物体の3D寸法と任意尺度のカメラ移動量を混ぜないため、右側へ物体を蓄積しません。**
姿勢のある画像でも、物体推論は独立したカメラローカル座標で実行します。
追跡・融合・3D視線ターゲット推定・実寸点群入力は拒否します。
`--neon_gaze`による2D視線表示は使用できます。
この経路では軌跡の有無によらずBoxer推論をFloat32に揃えます。

新しく復元する場合は`--colmap_source`を省略します。
`--prepared`も省略すれば画像の歪み補正・縮小から実行します。
学習特徴の抽出・対応付けにはCUDAと保存済みのLightGlue実行環境、
幾何検証・復元には既存のPyCOLMAP実行環境が必要です
（`output/neon_lightglue_runtime`、`output/neon_sfm_runtime`）。
ニューラル処理はCUDA Float32です。現在のPyCOLMAP配布物にはCUDA版Ceresがないため、
BAは従来のSfM検証と同じCPU設定を明示的に使います。CPUへ推論を切り替えることはありません。
専用IMUノイズ値やORB-SLAM3の初期化は不要です。

| 追加オプション | 既定値 | 内容 |
|---|---|---|
| `--pose_backend` | `orbslam3` | `orbslam3` / `colmap` |
| `--colmap_source` | 未指定 | 保存済み`mapping_auto_*`または数値モデルディレクトリ。省略時は新規復元 |
| `--colmap_map_policy` | `largest` | 選択範囲で最大の一地図、または`separated`で独立地図ごとに再生 |
| `--colmap_map_id` | 未指定 | 元モデルIDを明示選択。品質不合格は拒否 |
| `--nonmetric_preview` | 無効 | COLMAPの未検証尺度・重力を明示的に許可。COLMAP経路では必須 |
| `--colmap_scale_m_per_unit` | 未指定 | 位置の尺度候補。指定だけでは実寸追跡を許可しない |
| `--colmap_align_gravity` | 無効 | ネイティブIMUから共通回転の候補を求める。整合しても未検証扱い |
| `--poses-only` | 無効 | 軌跡生成・変換・評価まで。従来の`--vio-only`と同じ指定 |

`--pose_backend`、`--colmap_source`、`--colmap_map_policy`、`--colmap_map_id`、
`--colmap_scale_m_per_unit`、`--nonmetric_preview`には、アンダースコアを
ハイフンに置き換えた別名もあります。
`--boxer-only`で保存済みCOLMAP結果から推論を再開する際も、
`--nonmetric_preview --require_gpu --point_cloud off`を付けます。

複数地図の場合は`--colmap_map_policy separated`を指定します。
重複画像は「選択範囲で登録画像数が多い地図、同数なら元モデルIDが小さい地図」へ一度だけ割り当てます。
地図IDと画像欠損の境界で区切り、未登録画像の姿勢は補間しません。
Viewerは地図切替時に独立座標を切り替え、地図間を線や物体追跡で接続しません。
同一地図が再登場する場合も元の世界IDを保持します。

変換器単体は次のように使用できます。

```bash
python convert_colmap_trajectory.py \
  --prepared output/neon_colmap_good_02/prepared \
  --source output/neon_pose_comparison_02/Q-good_C0_01/mapping_auto_900 \
  --output output/neon_colmap_conversion_02 \
  --map-policy largest
```

元の`camera_to_world`を再度逆変換せず、3×4形式は4×4へ補完します。
元フレームID・int64時刻・回転・併記CSVの四元数を照合します。
NPZの`geometry_metadata`と`pipeline.json`に尺度、重力、モデル、変換、元データのSHA-256を保存します。
`visual_pose_frames`と`metric_pose_frames`を分け、現在の後者は0です。
任意尺度の速度は`speed_visual_units_s`で記録し、m/sや位置精度とは扱いません。
既存の`--prepared`も指定した`--start_n` / `--max_n`で抽出し、
900枚を指定したのに録画全体を母数にする誤集計を防ぎます。
失敗時はCOLMAP経路の`failure.json`に理由を記録し、作成済みの結果を保全します。

実寸追跡への移行には、尺度に使っていない別の距離等による検証、重力・外部校正・同期・
地図品質の確認が残っています。今回のCLIでは尺度候補を指定しても`metric=False`を維持します。
この接続の動作確認を、実寸3D追跡や位置精度の改善と同一視しないでください。

### 2026-10-21の実データ確認

以下はいずれも`sample_data/neon_Htou`を使用した結果です。

| 入力・方針 | 視覚姿勢 | 率 | 表示する独立地図 | 姿勢なし | 推論・Viewerのフレーム数 |
|---|---:|---:|---:|---:|---:|
| Q-good 9988〜10887 / largest | 694 | 77.11% | 1 | 206 | 900 |
| Q-hard 25834〜26733 / largest | 204 | 22.67% | 1 | 696 | 900 |
| 同じQ-hard / separated | 886 | 98.44% | 12 | 14 | 900 |

これらの**検証済み実寸姿勢はすべて0枚**です。表は位置精度の比較ではありません。
元のQ-hardの13モデルのうち一つは、重複フレームの割当後に残る画像が0枚となるため、
表示する世界は12個です。地図を接続した結果ではありません。

Boxerの物体推論は良好・難区間の各900枚でCUDA Float32を使用しました。
separatedの確認には、同じ難区間のカメラローカル推論結果を再利用しました。
視覚姿勢はニューラル推論に入れず、元画像・元時刻・校正・IMUによるローカル傾きが
一致するためです。各地図へ物体を変換・融合した再利用ではありません。
ViewerはNVIDIA GeForce RTX 3070 Laptop GPUで各900枚を順に描画し、
先頭へのシーク、地図切替、追跡が更新されないこと、欠損・別地図をまたぐ軌跡線がないことを確認しました。
保存先は次のとおりです。

```text
output/neon_colmap_good_preview_01/
output/neon_colmap_hard_preview_01/
output/neon_colmap_hard_separated_01/
```

すぐに複数地図版を再生するコマンドです。

```bash
python view_tracker.py \
  --input sample_data/neon_Htou \
  --output_dir output/neon_colmap_hard_separated_01/viewer \
  --require_gpu --autoplay
```

既存モデルを指定しない新規経路も、9988〜10027の40枚で実行し、
画像前処理→CUDA Float32のSuperPoint / LightGlue→DB構築→COLMAP→変換→評価が完了しました。
最大地図に40/40枚を登録しました。BAは明示的なCeres CPUです。
保存先は`output/neon_colmap_fresh_40_01/`です。
この40枚の結果から、全31,405枚の連続性や取得率を推定することはできません。

既定のORB経路も同じ9988〜10887の900枚で実行し、
3セッション・2実寸区間・187枚（20.78%）を出力しました。
区間は10162〜10244の83枚と10305〜10408の104枚です。
従来のバイナリ、校正、喪失時の停止方針、出力形式を保持し、
`--prepared`の選択範囲が900枚であることを確認しました。
並列処理による試行間変動があるため、枚数の完全一致を回帰条件にはしていません。
保存先は`output/neon_orb_colmap_regression_01/`です。

重力補正候補の確認では、良好区間694枚の重力方向の広がりはp95で2.959度でした。
候補回転は作成できましたが、外部校正・同期が独立検証されていないため、
`gravity_aligned_unverified`を維持しています。
画像間回転とジャイロの残差p95は良好区間2.653度、難区間最大地図4.573度です。
これらは内部の整合性診断であり、外部基準に対する姿勢精度ではありません。

回帰試験は`.venv/bin/python -m pytest -q tests`で586件成功・12件スキップでした。
GPU推論・NVIDIA描画は上記の実データ試験で別途確認しています。
元フレーム対応、変換、尺度・品質による拒否、直接融合の拒否、地図切替、
途中失敗時の成果物保全を含みます。`output/`内の外部ライブラリのテストは収集対象に含めません。

検証ログは`output/neon_colmap_validation/`、各Viewerの結果は
各実験出力の`viewer_validation.json`、スクリーンショットは`viewer_frame_*.png`に保存しています。
既存の98.11%基準DBは変更せず、SHA-256は
`c343992f366b2b087a83bb003cc4f06065b3c93388ca6554ae895edf7ea90e1f`のままです。

## 実行順序（既定のORB-SLAM3）

`prepare_neon_vio.py` → `mono_inertial_neon` →
`convert_neon_trajectory.py` → `run_boxer.py` → 動画・CSV確認です。
`run_neon_pipeline.py` はこの順序で全処理を実行します。

初回またはC++コード更新後は `bash scripts/build_neon_vio.sh` を実行してください。
Neon reader、SciPy、PyTorch、Boxerの重み、FFmpegが必要です。

```bash
cd /home/ubint-lab/boxer
source .venv/bin/activate

python run_neon_pipeline.py \
  --input sample_data/neon_Htou \
  --output output/neon_Htou_run_01 \
  --width 800 \
  --experimental-noise \
  --require_gpu \
  --skip_viz

python view_tracker.py \
  --input sample_data/neon_Htou \
  --output_dir output/neon_Htou_run_01/viewer \
  --require_gpu \
  --autoplay
```

`--max_n` を省略すると記録全体（既定上限999999フレーム）を処理します。
動作確認なら `--max_n 30` などで入力範囲を限定できます。
`--skip_viz` は推論時のJPEG/MP4生成を省略し、Viewer用のCSVは保存します。
前処理・軌跡推定・軌跡変換・軌跡評価・物体検出/3D推論・Viewer用統合の
工程名と進捗率をターミナルに表示し、出力先の`progress.log`にも保存します。
長い子プロセスは約5秒ごとに更新し、モデル読み込み中など進捗通知がない間も
状態を表示します。推論の進捗率は選択した全フレームに対する処理済み数で、
実行中の区間の進捗率も併記します。軌跡推定はセッションをまたいで集計します。
変換・評価・統合などの進捗率は処理件数に基づく値で、残り時間の割合ではありません。
細かな進捗を取得できない処理は開始0%・完了100%を表示します。
工程別の詳細ログ（`prepare.log`、`session_*/slam.log`、`boxer_*.log`等）も保存します。

```text
[物体検出・3D推論] 22.8% (7174/31405) 処理中 | 軌跡なし区間 6/29 | 区間 75.4% (1366/1812)
```

この環境での解析はGPUを必須とし、`--require_gpu` を指定します。
CUDAを利用できなければ処理開始前に停止し、CPUへ自動で切り替えません。
`run_boxer.py` 単体でも同じオプションを使えます。`--force_cpu` は指定しません。
Viewerにも `--require_gpu` を指定します。CUDAとOpenGL描画の両方を確認し、
`llvmpipe`などのCPU描画では停止します。LinuxではNVIDIA GLXオフロードを
選択します（明示済みの環境変数は保持）。起動ログの `[GPU] OpenGL renderer:`
で使用GPUを確認できます。
区間ごとの融合も必要なら `--fuse` を追加します。

特徴量キャッシュはCPU上の`float32`で保存し、OWLv2へ渡すときにだけ
推論モデルの数値型へ変換します。対応GPUの`bfloat16`推論は継続します。
Tracker・融合・Viewerでラベルを比較する特徴量は`float32`へ揃え、
単位長に正規化します。既存の`bfloat16`キャッシュも読み込み時に変換するため、
キャッシュ削除や解析のやり直しは不要です。CSVの小文字ラベル（例：`ipod`）も
元のキャッシュキー（`iPod`）を利用できます。キャッシュにないラベルを新規計算
した場合にも同じ数値型を使います。

Neonの軌跡なしフレームでは、利用可能なネイティブIMU姿勢から重力方向を求め、
BoxerNetの推定用座標軸へ反映します。IMUサンプルの間はSLERPで補間しますが、
計測範囲外・無効な姿勢・50 msを超える欠測区間は補間せず、従来の水平仮定へ戻します。
世界位置は生成せず、右画面の3D構築・追跡・融合には従来どおり有効なカメラ軌跡が必要です。
CSV上のカメラローカル座標とカメラ姿勢は維持するため、既存Viewer入力も利用できます。
軌跡ありフレームの重力はSLAM側の世界座標を使います。

パイプラインは前処理の`manifest.json`を`run_boxer.py --neon_imu_config`へ自動で渡し、
画像のprimary/aux時計・IMU時刻補正・カメラとIMUの外部校正を引き継ぎます。
`run_boxer.py`単体では、このオプションを省略するとprimary時計・0 ms補正・
メーカー公称の外部姿勢を使います。`boxer_input.json`の`local_gravity`に使用数と設定を保存します。
Neonの点群深度は画素ごとに最も手前の正のカメラZを選択してから、パッチ内で平均します。
同じ画素に投影された手前の物体と奥の壁の距離を平均し、実在しない奥行きを入力することを防ぎます。
これは新しく3D推論する際に適用されます。保存済み3D CSVや`--cache3d`による再利用には適用されません。

`--input neon_import_1` は `sample_data/neon_import_1` の直下にある単一の
記録フォルダも解決します。複数の記録がある場合は対象フォルダを直接指定します。
出力先は空である必要があります。再試行では新しい名前を使ってください。

過去のneon_import_1で使用した+60 msは、その記録用の**試験設定**です。
neon_Htouなど別の記録へ自動で引き継ぎません。
既定の時刻補正は0 msです。校正済みの値があれば `--imu-offset-ms`、
`--imu-noise noise.json`、`--extrinsics extrinsics.json` で指定してください。
`--experimental-noise` は未校正のノイズ設定を明示的に許可します。

`--vio-only` で軌跡と点群までを実行できます。保存済みの結果からBoxerだけを
実行する場合は同じ出力先に `--boxer-only` を指定します。

```bash
python run_neon_pipeline.py --input sample_data/neon_Htou \
  --output output/neon_Htou_run_01 --boxer-only --require_gpu --skip_viz
```

`--prepared PATH` を指定すると既存の前処理を再利用します。その場合、
フレーム範囲・解像度・時刻補正・IMU設定は前処理側の設定が使われます。
`--labels chair,table,monitor` で対象物体を限定できます。

`--gaze_targets --point_cloud off` を追加すると、点群をBoxerへ入力せず、
Neon視線と推定3Dボックスの交点から注視位置の候補もCSV・動画へ出力します。
`--neon_gaze` は視線方向と画像表示のみ、`--gaze_max_delta_ms` は視線の
時刻許容差（既定10 ms）です。
`--point_cloud auto/on/off` は有効なSLAM軌跡区間のBoxer入力に適用します。
軌跡のない区間は自動で点群なしの独立したフレーム座標として推論します。
`--point_cloud on` の点群必須条件は、有効な軌跡区間にだけ適用します。
座標と出力列の説明は[視線・注視位置の手順](neon_gaze_alignment.md)を参照してください。

## 出力と区間の扱い

```text
output/neon_run_01/
  prepared/                  補正画像、IMU、同期・画像品質診断
  session_000/slam/          states.csv、camera_trajectory.csv、map_points.csv
  session_000/geometry/      geometry_segment_*.npz、report.json
  session_001/...            追跡失敗後の再開セッション
  boxer_000/<recording>/     3D検出・追跡・融合CSV、MP4
  frame_only_000/<recording>/ 未追跡フレームの独立した3D推定、MP4
  viewer/<recording>/        全フレームを時刻順にまとめたViewer入力
  geometry_evaluation_000.json IMU整合性と、基準がある場合の誤差評価
  pipeline.json             有効区間、点群数、カバー率、結果パス
```

マップリセット前に初期化済みの区間を保存し、残りの画像は別プロセスで
再開します。保存時にはSLAMのバックグラウンド最適化の終了を待ちます。
一時的な追跡喪失（`RECENTLY_LOST`）だけでは終了せず、初期化済みマップを
保持したままORB-SLAM3の復帰処理を続けます。内部でリセットが要求された場合、
IMU初期化が失われた場合、または一時喪失以外の状態になった場合は保存して終了します。
復帰待ちに独自の無制限な猶予は設けず、ORB-SLAM3の既存の喪失タイムアウトを使います。
復帰に失敗した場合は、待機中に読んだ未追跡画像を次のセッションへ戻します。
再初期化に使える画像を復帰待ちのために消費してしまうことを防ぎます。
成功済みの軌跡は重複して採用しません。待機画像の再処理分だけ処理時間が増える
場合があり、軌跡推定の進捗は次セッションの再開位置へ戻る場合があります。
`session_*/slam/run.json` と `pipeline.json` の各セッションに
`recovery_attempts`（復帰待ち開始回数）、`recovered_episodes`（追跡復帰回数）、
`recovery_frames`（復帰待ちの画像数）、`recovery_replay_frames`
（次のセッションで再処理する未追跡画像数）を保存します。
変換は実行時と最終マップの両方でIMU初期化済み、かつ追跡成功のフレームだけを
採用します。未初期化・追跡失敗フレームを補間して埋めることはありません。
有効な軌跡を長さだけで取りこぼさないよう、`--min-frames` の既定値は2です。
速度・回転差の評価には連続する2姿勢が必要なため、孤立した1姿勢は採用しません。
以前の30フレーム以上という長さ制限は `--min-frames 30` で再現できます。
短い独立マップが増えるため、Viewerの世界切り替えや区間別推論の回数も増えます。
これは短い有効区間の採用範囲の変更であり、未初期化フレームの軌跡を生成する処理ではありません。
区間ごとに原点が異なるので、独立したBoxer処理として出力します。
この変更はC++アダプターの再ビルド後、新しく実行する軌跡推定から有効です。
既存CSV・Viewer結果を書き換えるものではありません。ORB-SLAM3の軌跡推定は
CPU実装で、`--require_gpu` が検証するBoxerのCUDA推論・Viewer描画とは別です。

選択した入力フレームは、有効な軌跡区間に含まれないフレームも自動で推論します。
IMUの前後サンプルがなくSLAMの前処理から除外された画像も、検出・再生には含めます。
`--frame-fallback` は過去のコマンドとの互換用に受け付けますが、指定は不要です。
軌跡なしの座標系は各フレームで独立した
`per_frame_assumed_upright` であり、カメラ位置が計測できたという意味では
ありません。区間ごとの元データは別のディレクトリへ保存し、推論後に
`viewer/<recording>/` へ検出CSV・視線CSV・フレーム表をまとめます。
`pipeline.json` の `coverage` は選択した全画像に対する有効な世界軌跡の割合です。
`frame_only_frames` と `total_estimated_frames` は独立して集計します。

疎点群はSLAMが三角測量した点であり、深度センサーの測定ではありません。
無効な点を除き、3キーフレーム以上で観測された点を出力します。
カメラ姿勢と点群を同じ量だけ平行移動し、同じメートル単位のZ-up座標系に
揃えます。Boxerは利用可能な点群を深度入力として使います。
静的マップの点を「現在のフレームで見えている」という証拠には使いません。

主なファイルは `boxer_3dbbs.csv`（各フレームの検出）、
`boxer_3dbbs_tracked.csv`（追跡履歴）、`boxer_3dbbs_fused.csv`
（`--fuse`指定時の静的物体）、`boxer_viz_final.mp4` です。
追跡と融合は別々の処理として順番に実行します。

## Viewerでの視線・注視点再生

処理終了時に、全フレーム再生用の `view_tracker.py` コマンドを表示します。
同じコマンド列は `pipeline.json` の `viewer_command` に保存します。
出力先は `--output_dir <pipeline出力>/viewer` で、軌跡・区間範囲の再指定は不要です。

左・中央の画像は元の全フレームを連続再生します。軌跡がないフレームでも
2D・3D検出を表示し、右の世界モデルは保持します。検出0件の画像も再生します。
SLAM区間が変わると、右画面の点群・軌跡・Trackerをその区間の世界へ切り替えます。
独立した世界同士を位置合わせしたり、Trackを共有したりはしません。
逆方向・非連続のシークでは、従来どおり移動先から追跡を開始します。

`--gaze_targets` で推論すれば、画像上の視線・推定注視候補も再生します。
右画面には有効な姿勢がある時刻だけ視線・推定注視点を表示します。
`--point_cloud off` をViewerに指定すれば、点群だけを非表示にした比較もできます。
`viewer/` の入力は区間のCSV・メタデータ・姿勢NPZも参照するので、pipeline出力全体を
保管してください。融合には `boxer_000` などの区間別出力を使います。
軌跡が全く得られない場合も、全フレームの検出と画像再生を完了します。
詳細なコマンドは[Viewerでの3D表示](neon_gaze_alignment.md#5-viewerで視線注視位置を3d表示)を参照してください。

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
- 追加の+80 ms、800幅・特徴点4000、CLAHEコントラスト補正＋特徴点4000の
  試験でも有効な30フレーム以上の区間は得られませんでした。
  既存の117フレームの結果を採用し、失敗を隠すための閾値緩和はしていません。

これらの修正は全618フレームの連続した軌跡や実寸精度を保証しません。
記録時のぼけ・白飛びはコードで元に戻せず、IMUノイズと同期・外部パラメータの
校正、外部基準軌跡による評価が残ります。点群付き3D物体推定の出力成功と、
位置・サイズの精度検証は区別してください。

## neon_Htouの軌跡取得改善の検証（2026-10-18）

今回採用した変更は、一時喪失からの復帰待ち・失敗時の未追跡画像の再処理と、
有効区間の最小長の既定値を30から2フレームへ変更する処理です。
IMU初期化済み・追跡成功・最終マップもメートル単位という判定は維持しています。

既存の全31,405フレーム分のSLAM出力を再変換した結果、採用可能な軌跡は
3,616フレーム（11.51%、28区間）から4,009フレーム（12.77%、54区間）に増えました。
増加した393フレームは短い有効区間の追加採用であり、新しく推定できた姿勢ではありません。
既存3,616フレームの姿勢・点群は一致し、追加した短い区間の最大速度は約0.77 m/sでした。
これは内部整合性の確認であり、実世界での絶対精度を保証するものではありません。

復帰処理は、幅800、特徴点2000、時刻補正0 ms、`PACE=1`、
最小区間長30を両方に揃え、次の元フレーム範囲で比較しました。

| 元フレーム範囲 | 修正前の有効フレーム | 復帰処理修正後の有効フレーム |
| --- | ---: | ---: |
| 1–900 | 262 | 164 |
| 9,988–10,887 | 355 | 628 |
| 15,001–15,900 | 0 | 0 |
| 30,001–30,900 | 0 | 0 |
| 合計3,600フレーム | 617（17.14%） | 792（22.00%） |

所要時間は約406.8秒から416.6秒でした。各条件1回の試行で、SLAMの並列実行に伴う
初期化の変動を含みます。この4区間では復帰成功は0回だったため、上表の取得率の差を
復帰処理による改善と断定できません。失敗した復帰待ち2回では、それぞれ50フレームを
次セッションへ戻し、採用姿勢の重複がないことを確認しました。

別途、元フレーム9,988–10,317の330画像のうち10,238だけを黒画像に置き換え、
一時遮蔽を模擬しました。遮蔽までの追跡状態列は両条件で一致し、修正前は終了、
修正後は10,239で復帰しました。有効軌跡は180から259フレームへ増加し、
遮蔽画像自体は採用されていません。両条件とも最小区間長30です。

初期化前の最終履歴を採用する案も試しましたが、最大約138 m/sの不自然な位置変化を
確認したため変更を戻しました。中盤900フレームでの+80 ms時刻補正の試験も、
30フレーム以上の有効区間は0のままで、既定値には採用していません。

ビルドは `cmake --build output/neon_vio_build --target mono_inertial_neon --parallel 2`、
回帰テストは次のコマンドで44件成功しました。

```bash
.venv/bin/python -m pytest tests/test_neon_tracking_policy.py tests/test_neon_vio.py \
  tests/test_neon_pipeline.py tests/test_neon_pipeline_replay.py tests/test_neon_progress.py -q
```

実行コマンド、比較用スクリプト、ログ、集計JSONはローカルの
`output/trajectory_recovery_validation/` に保存しています。
採用範囲の全体検証は `short_intervals/result.json` と
`short_intervals/geometry_validation.json`、復帰処理の比較は `comparison_final.json`、
遮蔽試験は `occlusion_validation.json` を参照してください。
`reconverted_existing/` は不採用にした初期化前履歴の試験出力です。

100%に近い取得率には達していません。新しい復帰処理で全31,405フレームのSLAMを
再実行した取得率と、独立した基準軌跡による精度は未測定です。
既存のViewer用出力は上書きしていません。短い区間の採用増加により、
今後の全解析ではモデルの区間別読み込み回数も増えるため、推論全体の時間は未評価です。

## センサと3D推定入力の監査・修正（2026-10-18）

調査対象はNeonのネイティブRGB・IMU・視線、前処理、SLAM出力、BoxerNetへの
画像・点群・重力入力、追跡と融合の世界区間、CSVとViewerです。
LiDARは未計測との申告があり、現在の記録にはLiDARによる深度・基準軌跡はありません。
別途保存されていたCOLMAP結果は`neon_import_1`の任意スケールの復元で、
`neon_Htou`のメートル単位の世界地図としては使用していません。

`neon_Htou`にはRGB 31,405フレームとIMU 329,764サンプルがあります。
実際のRGB間隔の中央値は100.027 ms、IMUは9.628 ms、最大間隔は19.205 msです。
カメラからIMUへの公称回転-102°、mmからm、角速度の度からrad、加速度のgからSIへの
変換は既存実装と公式資料の整合を確認しました。
[Pupil Labsの座標変換](https://docs.pupil-labs.com/alpha-lab/imu-transformations/)も参照してください。
ネイティブファイルの四元数はwxyzですが、使用中の公式Python readerの`rotation`はxyzwです。

次の2点を修正しました。

1. 軌跡のないフレームで、利用可能なIMU姿勢を使わずカメラが水平だと仮定していた入力。
   IMUの重力方向をカメラローカル座標へ変換し、モデルのvoxel座標軸だけを補正します。
   磁北の方位は使わず、移動位置も推定しません。既存の有効世界姿勢は変更しません。
2. 点群の投影で、同じ画素に前景・背景の点が来ると奥行きを平均していた入力。
   Neonでは最も手前の深度を選ぶz-bufferへ変更しました。
   他のデータセットは従来の集約が既定です。静的地図を「現在観測した点」とする扱いもしていません。

旧解析の軌跡なし27,789フレームのうち27,788フレームでは、IMUの前後サンプルから
重力方向を求められます。水平仮定とIMU重力の角度差は中央値17.37°、95%値36.19°でした。
これは世界位置を取得できたフレーム数の増加ではありません。

| 検証対象 | 修正前 | 修正後 |
| --- | ---: | ---: |
| 軌跡なし・静止に近い1,032フレームでの加速度と重力方向の差、中央値 | 21.91° | 2.36° |
| 同上、95%値 | 49.63° | 7.62° |
| 軌跡なし16画像・86ボックスの平均2D再投影IoU | 0.5874 | 0.6149 |
| 軌跡あり16画像・82ボックスの平均2D再投影IoU | 0.5475 | 0.5496 |

静止に近い区間の抽出条件は角速度ノルム5°/s未満、加速度ノルム1 gとの差0.05 g未満です。
IMU姿勢の融合には加速度も使われるため、重力方向の差は独立した精度評価ではありません。
再投影IoUも入力した2D検出との整合性であり、独立した物体位置・寸法の誤差ではありません。
軌跡あり画像の中央値IoUは0.6223から0.6202で、個々の物体すべての改善を保証しません。

点群の監査では、425フレームの投影画素1,081,176個のうち91,643個（約8.5%）に
複数の深度が重なっていました。平均深度と最も手前の深度との差は中央値0.21 m、
95%値5.27 m、最大500.42 mでした。これは点群投影方法間の差であり、LiDARを基準に
測った実世界の誤差ではありません。最も手前の面を選ぶ処理自体は合成データでも検証しました。

GPU比較はRTX 3070 Laptop、同一チェックポイント・画像・キャッシュ済み2D検出・
カメラ姿勢、BF16推論で実行し、32フレームすべて有限の3D出力を確認しました。
さらに、実データ30フレームを`run_neon_pipeline.py --require_gpu --skip_viz --gaze_targets`
で処理し、30フレームの統合CSV、29フレームのIMU重力使用、視線出力を確認しました。
実際のNVIDIA OpenGL Viewerで全30フレームと逆方向シークも確認しました。
リポジトリ全体の`PYTHON=.venv/bin/python bash scripts/ci.sh`は516成功・12スキップでした。
ログ・測定スクリプト・JSONは`output/neon_geometry_audit/`に保存しています。

推論時間のGPU比較はウォームアップの影響を含む小規模なもので、高速化の主張には使用しません。
今回の修正で全31,405フレームの3D再推論とSLAM再解析は行っておらず、既存の全解析結果は保持しています。
軌跡採用可能率12.77%の改善や、位置・寸法の絶対誤差の改善率は未測定です。

### LiDARを今後利用する場合

同じ環境のメートル単位の点群、座標軸・単位、Neon画像との対応関係が必要です。
固定したセンサ構成で同時計測する場合は、時刻同期とLiDARからカメラへの外部姿勢も必要です。
LiDARを別に手持ちで計測する場合、そのLiDARの移動姿勢をそのままNeonの移動姿勢にはできません。
事前の室内地図からNeonのカメラ位置を求める処理が別途必要です。
独立したSLAM区間にはそれぞれの地図への位置合わせが必要で、同じ点群を未登録のまま
全区間へ追加してはいけません。登録済みの姿勢と点群は既存の`--neon_geometry`形式で入力できます。
実際のLiDARデータによる校正・絶対精度・欠損区間の復元は、計測後の検証事項です。

## 評価方法

有効な区間ごとに `geometry_evaluation_*.json` を自動生成します。
ジャイロを実際の画像間隔で積分してカメラの相対回転と比較し、速度統計と
疎点群の画像内投影数も記録します。これは同じIMUを使った内部整合性の検査で、
絶対位置・距離・サイズの精度を測ったものではありません。
117フレームの採用軌跡ではジャイロ回転残差の中央値が約0.146°、95%値が
約0.410°でした。

提供された `neon_import_1` と `neon_import_2` は元のセンサー記録であり、
同じ記録の基準軌跡や実測した物体寸法は含まれていません。
基準がない場合、評価ファイルの `reference_accuracy` は測定不能と明記します。

独立に計測したNeonシーンカメラ軌跡がある場合は、
`timestamps_ns`（元の正確なint64 ns）と `T_world_camera`（メートル単位の
剛体変換）を含むNPZを `--reference` に指定できます。
基準には評価する全フレームの正確な時刻が必要です。
評価は最初のカメラ姿勢で座標原点・向きを合わせ、位置RMSEと回転誤差を計算します。
スケールはフィットせず、メートル単位のスケール誤差が隠れないようにします。

```bash
python evaluate_neon_geometry.py \
  --prepared output/neon_run_01/prepared \
  --geometry output/neon_run_01/session_000/geometry/geometry_segment_000.npz \
  --reference /path/to/independently_measured_reference.npz \
  --output output/neon_run_01/reference_evaluation.json
```

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
