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
    def __init__(self, classes, multi_label):
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

    def _clean(self):
        """
        labels内のテキストをクリーニングする
        """
        self.classes = [clean_text(class_) for class_ in self.classes]

    def get_classes(self):
        """
        labelsを取得する
        """
        return self.classes

class Output():
    """
    1つ1つのOutputを扱うためのクラス
    """
    def __init__(self, label, classes, str_=None, source_node=None):
        """
        Args:
            label (str): 出力ラベル
            classes (Classes): Classesクラス
            str_ (str): 出力文字列。strから変換された場合はうまくlabelsを取得できない場合があるので確認用に保持しておく
            source_node (list[dict]): predict時に参照したindexデータのリスト
        """
        self.label = label
        self.classes = classes
        self.multi_label = classes.multi_label
        self.str = str_
        self.source_node = source_node

    def get_index(self):
        if self.multi_label:
            return [self.classes.get_classes().index(label_elem) for label_elem in self.label]
        else:
            return None if self.label is None else self.classes.get_classes().index(self.label)

    def get_label(self):
        return self.label

    def get_str(self):
        if self.str is None:
            if self.multi_label:
                self.str = ", ".join(self.label)
            else:
                self.str = self.label if self.label is not None else ""
        return self.str

    def get_source_node(self):
        return self.source_node

    @classmethod
    def from_str(cls, str_, classes, source_node=None):
        """
        LLMのレスポンスからオブジェクトを出力する
        Args:
            str_ (str): 出力文字列(LLMのレスポンス)
            classes (Classes): Classesクラス
            source_node (list[dict]): predict時に参照したindexデータのリスト
        Returns:
            obj: Output
        """
        str_ = clean_text(str_)
        label = [label for label in classes.get_classes() if label in str_]
        if not classes.multi_label:
            label = label[0] if len(label) > 0 else None
        obj = cls(label, classes, str_, source_node)
        return obj

    @classmethod
    def from_index(cls, index, classes):
        """
        indexからオブジェクトを出力する
        Args:
            index (int or list[int]): index
            classes (Classes): Classesクラス
        Returns:
            obj: Output
        """
        if classes.multi_label:
            label = [classes.get_classes()[index_elem] for index_elem in index]
        else:
            label = None if index is None else classes.get_classes()[index]
        obj = cls(label, classes)
        return obj

    @classmethod
    def from_label(cls, label, classes):
        """
        labelからオブジェクトを出力する
        Args:
            label (str or list[str]): label
            classes (Classes): Classesクラス
        Returns:
            obj: Output
        """
        if classes.multi_label:
            label = [clean_text(label_elem) for label_elem in label]
        else:
            label = clean_text(label) if label is not None else None
        obj = cls(label, classes)
        return obj

class Outputs():
    """
    Outputsを扱うためのクラス
    """

    def __init__(self, outputs):
        """
        Args:
            outputs (list[Output]): Outputのリスト
        """
        self.outputs = outputs

    def get_indices(self):
        return [output.get_index() for output in self.outputs]

    def get_labels(self):
        return [output.get_label() for output in self.outputs]

    def get_strs(self):
        return [output.get_str() for output in self.outputs]

    def get_source_nodes(self):
        return [output.get_source_node() for output in self.outputs]

    def get_df(self):
        df = pd.DataFrame({
            "index": self.get_indices(),
            "label": self.get_labels(),
            "str": self.get_strs(),
            "source_node": self.get_source_nodes()
            })
        return df

    @classmethod
    def from_strs(cls, strs, classes, source_nodes):
        """
        LLMのレスポンスからオブジェクトを出力する
        Args:
            strs (List[str]): 出力文字列のリスト(LLMのレスポンス)
            classes (Classes): Classesクラス
            source_nodes (list[list[dict]]): predict時に参照したindexデータの集まり
        Returns:
            obj: Outputs
        """
        outputs = [Output.from_str(str_, classes, source_node) for str_, source_node in zip(strs, source_nodes)]
        obj = cls(outputs)
        return obj

    @classmethod
    def from_labels(cls, labels, classes):
        """
        ラベルからオブジェクトを出力する
        Args:
            labels (list[str] or list[list[str]]): ラベルのリスト
            classes (Classes): Classesクラス
        Returns:
            obj: Outputs
        """
        outputs = [Output.from_label(label, classes) for label in labels]
        obj = cls(outputs)
        return obj

    @classmethod
    def from_indices(cls, indices, classes):
        """
        indexからオブジェクトを出力する
        Args:
            indices (list[int] or list[list[int]]): indexのリスト
            classes (Classes): Classesクラス
        Returns:
            obj: Outputs
        """
        outputs = [Output.from_index(index, classes) for index in indices]
        obj = cls(outputs)
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
            "'Q_ex:'以降の情報について、同様の質問をした時の回答を'A_ex:'以降に示しています。\n"
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
        output_prompts = self.outputs.get_strs()
        prompt_template = (
            "Q_ex:{input}\n"
            "A_ex:{output}"
            )
        prompts = [
            prompt_template.format(i=i, input=input_prompt, output=output_prompt)
            for i, (input_prompt, output_prompt) in enumerate(zip(input_prompts, output_prompts))
            ]
        return prompts

    def get_whole_prompt(self):
        """
        examplesのpromptを取得する
        Return:
            prompt: str
        """
        prompts = self.get_each_prompts()
        prompts = [f"{i+1}.\n{prompt}" for i, prompt in enumerate(prompts)]
        prompt = "\n".join(prompts)
        prompt = self.prompt_template.format(context_str=prompt)
        return prompt