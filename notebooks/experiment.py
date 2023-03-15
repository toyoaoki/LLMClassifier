# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: exbert
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ZeroShot分類

# %%
from llmclassifier import Classifier
import pandas as pd
from langchain.llms import OpenAI
import os

# llmの設定
llm = OpenAI(model_name="text-davinci-003", openai_api_key=os.environ["OPENAI_API_KEY"])

# タスクの設定
task = """あなたはニュースのジャンル分類をしています。
ニュースのタイトルを読んで、そのジャンルを判断してください。"""

# ラベルの設定
classes = ["金融", "芸能", "食品", "音楽", "自動車", "経済", "政治", "スポーツ", "IT", "エンタメ", "科学", "国際", "地域", "健康", "教育", "お笑い", "事件", "その他"]

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
clsf = Classifier(llm=llm, task=task, classes=classes, multi_label=False)


# %%
# 結果の取得
# y = clsf.predict(inputs)
y_labels = clsf.predict(inputs, return_index=False)
df = clsf.predict(inputs, return_df=True)
# print(f"アノテーション結果のindex:\n{y}")
print(f"アノテーション結果のlabel:\n{y_labels}")
display(df)

# %% [markdown]
# # FewShot分類

# %%
X_train = pd.DataFrame([
        ["日銀が引き続き金利の引き上げを行うことを表明。"],
        ["ユーチューバー柴田が「世界のみんなを助けたい」と世界平和を訴える"],
        ["乳酸菌飲料の睡眠効果を調査。統計的に有意なことが確認された"],
        ["放課後音楽クラブがゲーム「パチモン」のED「パチモンしりとり」を発表"],
        ["MISSANが全ての自動車にリチウムバッテリーを搭載すると発表"],
        ["供給不足が解消しガソリン市場急落。前月比5%の値下がりを記録"],
    ],
    columns=["ニュースタイトル"])
y_train = ["政治", "お笑い", "地域", "国際", "事件", "その他"]

# %%
clsf.fit(X_train, y_train, few_shot=True)
y_labels = clsf.predict(inputs, return_index=False)
print(f"アノテーション結果のlabel:\n{y_labels}")

# %% [markdown]
# # LlamaIndexをつかった分類

# %%
clsf.fit(X_train, y_train, few_shot=False)
y_labels = clsf.predict(inputs, return_index=False)
print(f"アノテーション結果のlabel:\n{y_labels}")
