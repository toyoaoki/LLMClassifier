# LLMClassifier

ジェネラティブなLLMClassifierはLLMを使ったClassifierを構築します。

ジェネラティブLLMを分類器として使用し、システムなどに組み込む場合には以下の課題があります（2023/3/12現在）
- 出力形式 : 返答は文字列型であり、表記ブレも存在するため、そのままでは分類に使用できません。
- 学習 : LLMではFewShot分類は可能ですが、より多くのデータを活用しての分類はできません

上記の課題を以下の方法で解決したのがこのライブラリです。
- 出力形式 : テキストの標準化を行い表記ブレを無くした上で、分類結果を候補リストのインデックスとして出力します
- 学習 : Llama(GPT)Indexを用いてより多くの学習データを活用できるようにしました。

また、マルチラベル分類や、通常のFewShot学習にも対応しています。

# How to use

下記のコマンドでインストールしてください。
```bash
pip install git+https://github.com/toyoaoki/llmclassifier.git@main
```

詳しい使用法はnotebooks/以下に配置されたノートブックを参照してください。
なお、現状ノートブックはコード部分を.py形式で保存しているため、VSCodeなどで利用いただくか、jupytextなどで変換してご使用ください。

# TODO

今後は下記に対応する予定です。
- 出力方法の整理 : 出力形式を変えるごとにLLMにクエリを投げる形になっている課題を解決する
- 学習機能の改善 : 現状のGPT-indexだと類似度をうまく捉えられていないと思われるため改善する
- 別モデルによる学習 : Huggingface Transformerを用いて分類結果を学習し、モデルを出力する

