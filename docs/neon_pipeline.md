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
