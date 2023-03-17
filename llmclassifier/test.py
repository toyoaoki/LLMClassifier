import pandas as pd
from langchain.llms import OpenAI
import os

from classifier import LLMClassifier

# llmの設定
llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])

# タスクの設定
task = """あなたはニュースのジャンル分類をしています。
ニュースのタイトルを読んで、そのジャンルを判断してください。"""

# ラベルの設定
classes = ["金融", "芸能", "食品", "音楽", "自動車", "経済", "政治", "スポーツ", "IT", "エンタメ", "科学"]

# 入力データの設定
inputs = pd.DataFrame([
        ["日本銀行が金利を引き下げる"],
        ["ユーチューバーMUKAKINが「みんな有料会員になって」と自身のチャンネルで発言し話題に"],
        ["雲印、新製品「体いきいきヨーグルト」発売。体内フローラを整える"],
        ["麦津犬歯、新曲「入れもん」MV公開。曲に合わせて入れ物に残った匂いを嗅ぐ内容"],
        ["TOYBOTAが大幅な方針転換を発表。全ての水上自動車を電気自動車にする。来年2024年までに切り替えを完了する"],
        ["三月の鉄鋼市況は、前月比で鉄鋼製品の生産量は前年同月比で1.5％減少し、鉄鋼製品の販売量は前年同月比で1.1％減少した。"],
    ],
    columns=["ニュースタイトル"])

# annotatorの設定
clsf = LLMClassifier(llm=llm, task=task, classes=classes, multi_label=False, translate_to_english=True)

# 結果の取得
outputs = clsf.predict(inputs, return_wrapper=True)
df = outputs.get_df(add_original_label=True)
df.head()
