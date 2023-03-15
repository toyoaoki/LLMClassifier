import pandas as pd
from .util import clean_text


class Inputs:
    """
    Prompt内のinput部分をビルドするためのクラス
    """
    def __init__(self, data):
        """
        Args:
            data (dict, list[dict], pd.DataFrame): json, jsons or dataframe

        Raises:
            ValueError: data must be json, jsons or dataframe
        """
        # json型, jsons型かdataframe型を処理可能
        if isinstance(data, dict):
            self.jsons = [data]
        elif isinstance(data, list) and isinstance(data[0], dict):
            self.jsons = data
        elif isinstance(data, pd.DataFrame):
            self.jsons = data.to_dict(orient="records")
        else:
            raise ValueError("data must be json, jsons or dataframe")
        self._clean()

    def get_prompts(self):
        """
        self.jsons内のkey, valueをpromptに埋め込む。
        {key1:value1, key2:value2, ...} -> "{key1}{value1}¥n{key2}{value2}¥n..."
        Return:
            prompts: list[str]
        """
        prompts = []
        for json in self.jsons:
            prompt = []
            for key, value in json.items():
                prompt.append(f"{key}:{value}")
            prompt = "\n".join(prompt)
            prompts.append(prompt)
        return prompts

    def _clean(self):
        """
        jsons内のテキストをクリーニングする
        """
        self.jsons = [
            {clean_text(key): clean_text(value) for key, value in json.items()}
            for json in self.jsons]

    def get_df(self):
        """
        jsonsをdataframeに変換する
        """
        return pd.DataFrame(self.jsons)

    def get_jsons(self):
        """
        jsonsを取得する
        """
        return self.jsons

class Classes():
    """
    Prompt内のlabel部分をビルドするためのクラス
    """
    def __init__(self, classes, multi_label=False):
        """
        Args:
            classes (list[str]): ラベルのリスト
            multi_label (bool): マルチラベルかどうか
        """
        assert isinstance(classes, list), "classes must be list"
        self.classes = classes
        self._clean()
        self.multi_label = multi_label

    def get_prompt(self):
        """
        labelsを単一ラベル分類用のpromptに埋め込む。
        ["label1", "label2", ...] -> "label1, label2, ..."
        Return:
            prompt: str
        """
        prompt = ", ".join(self.classes)
        if self.multi_label:
            prompt = f"回答は「{prompt}」のいずれかの組み合わせで答えてください。','で区切って0個以上回答してください。"
        else:
            prompt = f"回答は「{prompt}」のいずれかで答えてください。"
        return prompt

    def extract_label_in(self, string):
        """
        string内に含まれるlabelまたはそのリストを抽出する

        Args:
            string (str): ラベルが含まれる文字列
        Rteturn:
            result: str or list[str]
        """
        string = clean_text(string)
        result = [label for label in self.classes if label in string]
        if not self.multi_label:
            result = result[0] if result else None
        return result

    def convert_to_indices(self, labels):
        """
        labelsをインデックスに変換する

        Args:
            other_labels (list[str] or str): ラベルのリスト
        Return:
            indices: list[int]
        """
        if self.multi_label:
            indices = [[self.classes.index(label_elem) for label_elem in label] for label in labels]
        else:
            indices = [None if label is None else self.classes.index(label) for label in labels]
        return indices

    def _clean(self):
        """
        labels内のテキストをクリーニングする
        """
        self.classes = [clean_text(class_) for class_ in self.classes]

class Outputs():
    """
    Outputを扱うためのクラス
    """

    def __init__(self, labels, classes):
        """
        Args:
            labels (List[str]): 出力ラベルのリスト
            classes (Classes): Classesクラス
        """
        self.labels = labels
        self.classes = classes

    def get_indices(self):
        return self.classes.convert_to_indices(self.labels)

    def get_labels(self):
        return self.labels

    def to_strs(self):
        """
        str化したものを出力する
        Returns:
            strs: list[str]
        """
        strs = [
            ", ".join(label) if isinstance(label, list) else label
            for label in self.get_labels()]
        return strs

    @classmethod
    def from_resposes(cls, responses, classes):
        """
        LLMのレスポンスからオブジェクトを出力する
        Args:
            responses (List[str]): LLMのレスポンス
            classes (Classes): Classesクラス
        Returns:
            obj: Outputs
        """
        labels = [classes.extract_label_in(response) for response in responses]
        obj = cls(labels, classes)
        return obj

class Examples():
    """
    Fewshotに使うExampleを扱うためのクラス
    """

    def __init__(self, inputs, outputs):
        """
        Args:
            inputs (Inputs): インプットデータ
            outputs (OutputData): アウトプットデータ
        """
        self.inputs = inputs
        self.outputs = outputs
        self.prompt_template = (
            "回答例は以下の通りです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "この回答例が使えるなら考慮に入れ、使えない場合は考慮に入れずに答えてください。"
            )
    
    def get_each_prompts(self):
        """
        それぞれのexampleのpromptを取得する
        Return:
            prompts: list[str]
        """
        input_prompts = self.inputs.get_prompts()
        output_prompts = self.outputs.to_strs()
        # prompt_template = (
        #     "入力:\n"
        #     "{input}\n"
        #     "出力:\n"
        #     "{output}"
        #     )
        prompt_template = (
            "「{input}」という情報の分類は「{output}」"
            )
        prompts = [
            prompt_template.format(input=input_prompt, output=output_prompt)
            for input_prompt, output_prompt in zip(input_prompts, output_prompts)
            ]
        return prompts

    def get_fewshot_prompt(self):
        """
        examplesのpromptを取得する
        Return:
            prompt: str
        """
        prompts = self.get_each_prompts()
        prompt = "\n".join(prompts)
        prompt = self.prompt_template.format(context_str=prompt)
        return prompt