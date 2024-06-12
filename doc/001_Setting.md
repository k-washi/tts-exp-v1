# 環境構築

```
docker compose build  \
    --build-arg uid=$(id -u) \
    --build-arg gid=$(id -g)


docker compose up -d
```

reference: [Ascender](https://github.com/cvpaperchallenge/Ascender/tree/develop)

## Docker内

ライブラリのインストール

```
poetry install
source .venv/bin/activate
pip install -e .
```

poetryでライブラリの依存関係を管理しpipで、自前のライブラリをeditable modeでimport可能にする。

## vscode extentions install

```
./.devcontainer/vscode_extentions_install_batch.sh 
```