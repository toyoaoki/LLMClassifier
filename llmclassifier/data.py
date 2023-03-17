import pandas as pd

from .util import clean_text, recursive_apply

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
        return self.apply(clean_text)

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

    def apply(self, func, *args, **kwargs):
        """
        inputにfuncを適用する
        Args:
            func (function): 適用する関数
            args: 関数に渡す引数
            kwargs: 関数に渡すキーワード引数
        Retrun:
            self (Inputs): 自身のインスタンス
        """
        df = recursive_apply(index=0)(func)(self.get_df(), *args, **kwargs)
        self.jsons = df.to_dict(orient="records")
        return self

    def save(self, path):
        """
        csvを保存する
        Args:
            path (str): 保存先のパス
        """
        self.get_df().to_csv(path, encoding="utf-8_sig", index=False)

class Classes():
    """
    Prompt内のlabel部分をビルドするためのクラス
    """
    def __init__(self, classes, original_classes, multi_label):
        """
        Args:
            classes (list[str]): ラベルのリスト
            original_classes (list[str]): 元のラベルのリスト
            multi_label (bool): マルチラベルかどうか
        """
        assert isinstance(classes, list), "classes must be list"
        self.classes = classes
        self.original_classes = original_classes
        self._clean()
        self.multi_label = multi_label

    def get_prompt(self):
        """
        labelsを単一ラベル分類用のpromptに埋め込む。
        ["label1", "label2", ...] -> "'label1', 'label2', ..."
        Return:
            prompt: str
        """
        classes = [f"'{class_}'" for class_ in self.classes]
        prompt = ", ".join(classes)
        if self.multi_label:
            prompt = ("Answer by choosing zero or more combinations from the following candidates enclosed in quotes and separated by commas.\n"
            "For each chosen candidate, output the exact same text, including capitalization, spacing, and special characters, as well as the quotation marks surrounding the candidate.\n"
            "The answer should be output with candidates separated by commas.\n"
            "The candidates are as follows:\n"
            "---------\n"
            f"{prompt}\n"
            "---------\n")
        else:
            prompt = ("Select one of the following candidates enclosed in quotation marks and separated by commas.\n"
            "For the chosen candidate, output the exact same text, including capitalization, spacing, and special characters, as well as the quotation marks surrounding the candidate.\n"
            "The candidates are as follows:\n"
            "---------\n"
            f"{prompt}\n"
            "---------\n")
        return prompt

    def _clean(self):
        """
        labels内のテキストをクリーニングする
        """
        self.classes = clean_text(self.classes, strict=True)

    def get_classes(self):
        """
        labelsを取得する
        """
        return self.classes

    def get_original_classes(self):
        """
        元のlabelsを取得する
        """
        return self.original_classes

class Output():
    """
    1つ1つのOutputを扱うためのクラス
    """
    def __init__(self, index, classes, str_=None, source_node=None):
        """
        Args:
            index (int or list[int]): 出力ラベル
            classes (Classes): Classesクラス
            str_ (str): 出力文字列。strから変換された場合はうまくlabelsを取得できない場合があるので確認用に保持しておく
            source_node (list[dict]): predict時に参照したindexデータのリスト
        """
        self.index = index
        self.classes = classes
        self.multi_label = classes.multi_label
        self.str = str_
        self.source_node = source_node

    def get_index(self):
        return self.index

    def get_label(self):
        label = recursive_apply(index=0)(lambda x: self.classes.get_classes()[x])(self.index)
        return label

    def get_str(self):
        if self.str is None:
            if self.multi_label:
                self.str = ", ".join(self.label)
            else:
                self.str = self.label if self.label is not None else ""
        return self.str

    def get_source_node(self):
        return self.source_node

    def get_original_label(self):
        label = recursive_apply(index=0)(lambda x: self.classes.get_original_classes()[x])(self.index)
        return label

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
        str_ = clean_text(str_, strict=True)
        index = [i for i, label in enumerate(classes.get_classes()) if label in str_]
        if not classes.multi_label:
            index = index[0] if len(index) > 0 else None
        obj = cls(index, classes, str_, source_node)
        return obj

    @classmethod
    def from_index(cls, index, classes, source_node=None):
        """
        indexからオブジェクトを出力する
        Args:
            index (int or list[int]): index
            classes (Classes): Classesクラス
        Returns:
            obj: Output
        """
        obj = cls(index, classes, source_node=source_node)
        return obj

    @classmethod
    def from_label(cls, label, classes, source_node=None):
        """
        labelからオブジェクトを出力する
        Args:
            label (str or list[str]): label
            classes (Classes): Classesクラス
        Returns:
            obj: Output
        """
        label = clean_text(label, strict=True)
        index = recursive_apply(index=1)(classes.get_classes().index)(label)
        obj = cls(index, classes, source_node=source_node)
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

    def get_df(self, add_original_label=False):
        df = pd.DataFrame({
            "index": self.get_indices(),
            "label": self.get_labels(),
            "str": self.get_strs(),
            "source_node": self.get_source_nodes()
            })
        if add_original_label:
            df["original_label"] = self.get_original_labels()
        return df

    def set_original_labels(self):
        self.original_labels = self.get_original_labels()

    def save(self, path):
        df = self.get_df()
        df.to_csv(path, encoding="utf-8_sig", index=False)

    def get_original_labels(self):
        return [output.get_original_label() for output in self.outputs]

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
            "Here is an example of an answer:\n"
            "For information after 'Q_ex:', the desired answer when the same question was asked earlier is shown after 'A_ex:'.\n"
            "---------\n"
            "{context_str}\n"
            "---------\n"
            "If this example answer can be used, consider it. If not, answer without considering it"
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