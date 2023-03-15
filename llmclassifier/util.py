import neologdn


def clean_text(text):
    """テキストのクリーニングを行う関数

    Args:
        text (str): クリーニング前のテキスト
    Returns:
        text (str): クリーニング後のテキスト
    """
    # neologdnで正規化
    text = neologdn.normalize(text)
    # 大文字のアルファベットを小文字に変換
    text = text.lower()
    return text