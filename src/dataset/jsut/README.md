# JSUT データセットのダウンロード

1. JSUTデータセットのダウンロード
2. JSUTラベルデータのダウンロード
3. データセットの形式を変換
4. 訓練用のファイルリストを作成
```
mkdir ./data
cd ./data
curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
unzip jsut_ver1.1.zip
git clone https://github.com/sarulab-speech/jsut-label

cd ../
python ./src/dataset/jsut/create_tts_dataset.py --jsut_dir ./data/jsut_ver1.1 --label_dir ./data/jsut-label --output_dir ./data/jsut --sr 22050

# 以下で訓練用のファイルを作成
python ./src/dataset/utils/create_namelist_files.py --input ./data/jsut/wav --output ./data/jsut --val_rate 0.1
```