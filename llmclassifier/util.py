import neologdn
import pandas as pd
from functools import wraps
from copy import deepcopy

def recursive_apply(index=0):
    """
    関数の第一引数にリスト、辞書、DataFrameが渡された場合に、再帰的に関数を適用するデコレータ
    """
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            def new_args(args, index, x):
                new_args = deepcopy(args)
                new_args = list(new_args)
                new_args[index] = x
                new_args = tuple(new_args)
                return new_args
            arg = args[index]
            if isinstance(arg, list):
                return [inner_wrapper(*new_args(args, index, x), **kwargs) for x in arg]
            elif isinstance(arg, dict):
                return {inner_wrapper(*new_args(args, index, k), **kwargs): inner_wrapper(*new_args(args, index, v), **kwargs) for k, v in arg.items()}
            elif isinstance(arg, pd.DataFrame):
                df = arg.copy()
                df.columns = df.columns.map(lambda x: inner_wrapper(*new_args(args, index, x), **kwargs))
                df = df.applymap(lambda x: inner_wrapper(*new_args(args, index, x), **kwargs))
                return df
            elif arg is None:
                return None
            else:
                return func(*args, **kwargs)
        return inner_wrapper
    return wrapper

@recursive_apply(index=0)
def clean_text(text, strict=False):
    """テキストのクリーニングを行う関数

    Args:
        text (str): クリーニング前のテキスト
        strict (bool): 標準化も含めた厳密なクリーニングを行うかどうか
    Returns:
        text (str): クリーニング後のテキスト
    """
    # neologdnで正規化
    text = neologdn.normalize(text)
    # 大文字のアルファベットを小文字に変換
    if strict:
        text = text.lower()
        text = (text
                .replace("\t", "")
                .replace("\n", "")
                .replace("\r", "")
                )
    return text