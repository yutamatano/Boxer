# Boxer の自動テスト（CI）

研究室PCのBoxerをVS Code Remote SSHで編集し、GitHubへpushすると、
GitHub ActionsがUbuntu 24.04 / Python 3.12 / CPU版PyTorch 2.6.0でテストする。
研究室PCのGPUやSSH接続、秘密鍵はCIには不要。

## 自動実行のタイミング

- すべてのブランチへのpush
- Pull Requestの作成・更新
- GitHub Actionsの「Run workflow」からの手動実行

同じブランチへ続けてpushすると、前のCIは中断して最新の変更をテストする。
CI失敗はすでに行われたpushを取り消さない。
mainへの取り込みを制限する場合は、GitHubのブランチルールで
Pull Requestと必須チェック「CPU tests (Python 3.12)」を設定する。

## 実行内容

1. Git管理中のPythonファイルの構文確認
2. Git管理中のシェルスクリプトの構文確認
3. 既存のpytestテスト一式（tests/）
4. テストログとJUnit XMLの保存（14日間）

座標変換、OBB、追跡、融合、CSV、Neon入力・軌跡変換、
Viewerの入力選択などを検証する。Neonテストでは合成録画を生成する。
Viewerの回帰テストはGUIモジュールを読み込むが、ウィンドウを開かない。

## モデル・データが必要なテスト

GitHubにはckpts/とsample_data/を送らない。
OWLv2の実モデルを使うテストは、チェックポイントがない場合に理由付きでスキップする。
モデルがあるローカル環境では従来どおり実行する。
transformersとの比較にはtransformersとモデルの両方が必要。

「モデルがない場合に適切なエラーが出るか」のテストは常に実行する。
このCIはGPU推論、実録画の解析精度、ORB-SLAM3のC++ビルド、
実際のGUI描画までは検証しない。

## 研究室PCでの使用

最初に今回のCI設定を取り込む。未コミット変更がある場合は内容を確認して整理する。

```bash
cd /home/ubint-lab/boxer
git status
git pull --ff-only origin main
```

以降はRemote SSHで編集し、必要なファイルだけコミットしてpushする。

```bash
git diff
git add <変更したファイル>
git commit -m "変更内容"
git push origin <作業ブランチ>
```

GitHubのリポジトリでActions → Boxer CI → 該当コミットを開く。
緑なら成功、赤なら失敗。失敗したステップのログで原因を確認し、
必要ならcpu-test-resultsからpytest.logとjunit.xmlを取得する。

## 同じチェックをローカルで実行

既存のBoxer仮想環境にpytestとscipy、Neon reader、Viewerの依存関係を用意する。

```bash
cd /home/ubint-lab/boxer
source boxer/bin/activate
bash scripts/ci.sh
```

Pythonを明示する場合：

```bash
PYTHON=/home/ubint-lab/boxer/boxer/bin/python bash scripts/ci.sh
```

CIと同じ依存関係を試す場合は、既存のGPU環境を変更せず、独立した仮想環境を作る。

```bash
python3.12 -m venv output/ci/venv
output/ci/venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
output/ci/venv/bin/python -m pip install -r .github/requirements-ci.txt
PYTHON="$PWD/output/ci/venv/bin/python" bash scripts/ci.sh
```

CPU版PyTorchは固定し、ほかの依存関係は対応範囲を指定している。
完全な依存関係の固定ではないため、将来の依存更新で失敗した場合は
Actionsのインストールログを確認する。

## GitHub側の設定

Settings → Actions → GeneralでGitHub Actionsを有効にし、
actions/checkout、actions/setup-python、actions/upload-artifactを許可する。
リポジトリがforkの場合、Actionsの初回有効化が必要になることがある。
ワークフローのGITHUB_TOKENはcontents: readのみを使用する。
CIから研究室PCへのコード反映や解析ジョブの起動は行わない。

公式資料：
- [PythonのテストをGitHub Actionsで実行](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [ワークフローを起動するイベント](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
