from langchain import PromptTemplate
from llama_index import GPTSimpleVectorIndex, Document, QuestionAnswerPrompt
import pandas as pd
import pickle

from util import clean_text, recursive_apply
from data import Inputs, Classes, Outputs, Examples
from translator import LLMTranslator

class LLMClassifier():
    """
    LLMを用いた分類器のクラス
    scikit-learnのインターフェースに準拠している。
    """

    def __init__(self, llm, task, classes, multi_label=False, translate_to_english=False):
        """
        Args:
            llm (langchain.llms): LLMオブジェクト
            task (str): LLに指定するタスク
            classes (list[str]): 出力するラベルのリスト
            multi_label (bool): マルチラベルかどうか
            translate_to_english (bool): 英訳を行うかどうか
        Raises:
            ValueError: data must be json, jsons or dataframe
        """
        self.llm = llm
        self.translator = LLMTranslator(llm)
        self.translate_to_english = translate_to_english
        self.task = self.translate_if_needed(task)
        original_classes = classes
        classes = self.translate_if_needed(classes)
        self.classes = Classes(classes, original_classes, multi_label)
        self.class_prompt = self.classes.get_prompt()
        self.awser_marker = "A:"
        self.last_prompt = self.awser_marker
        self.template = (
            "Q:{task}\n"
            "{classes}\n"
            "{input}\n"
            "{last}"
            )
        self.prompt_template= PromptTemplate(
            input_variables=["task", "classes", "input", "last"],
            template=self.template,
        )
        self.each_example_prompts = None

    def predict(self, X, return_wrapper=False):
        """
        分類を行う。LLMが行ったラベル付けの結果のリストを出力する。
        ラベルはlabelsのindexで表される。
        Args:
            X (dict, list[dict], pd.DataFrame): json, jsons or dataframe
            return_wrapper (bool): Outputsオブジェクトを返すかどうか
        Return:
            y: list[int] or list[list[int]] or list[str] or list[list[str]]
        """
        inputs = Inputs(X)
        inputs = inputs.apply(self.translate_if_needed)
        self.inputs = inputs
        input_prompts = inputs.get_prompts()
        self.prompts = [self.prompt_template.format(task=self.task, classes=self.class_prompt, input=input_prompt, last=self.last_prompt) for input_prompt in input_prompts]
        if hasattr(self, "llama_index"):
            index_responses = [self.llama_index.query(prompt, text_qa_template=self.qa_prompt) for prompt in self.prompts]
            source_nodes = [index_response.source_nodes for index_response in index_responses]
            responses = [index_response.response for index_response in index_responses]
        else:
            responses = [self.llm(prompt) for prompt in self.prompts]
            source_nodes = [self.each_example_prompts for _ in self.prompts]
        outputs = Outputs.from_strs(responses, self.classes, source_nodes)
        if return_wrapper:
            y = outputs
        else:
            y = outputs.get_indices()
        return y

    def fit(self, X, y, use_index=False):
        """
        few_shot = True の場合、Xとyをプロンプトに埋め込んで予測を行う
        few_shot = False の場合、GPT-indexを用いて予測を行う
        Args:
            X (dict, list[dict], pd.DataFrame): json, jsons or dataframe
            y (list[str], list[list[str]]): ラベルのリスト
            use_index (bool): llama_indexを用いるかどうか
        Returns:
            self
        """
        inputs = Inputs(X)
        inputs = inputs.apply(self.translate_if_needed)
        y = self.translate_if_needed(y)
        outputs = Outputs.from_labels(y, self.classes)
        examples = Examples(inputs, outputs)
        if use_index:
            texts = examples.get_each_prompts()
            documents = [Document(t) for t in texts]
            self.llama_index = GPTSimpleVectorIndex(documents)
            QA_PROMPT_TMPL = "{query_str}\n" + examples.prompt_template + "\n" + self.awser_marker
            self.qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)
            self.last_prompt = ""
        else:
            self.last_prompt = examples.get_whole_prompt() + "\n" + self.awser_marker
            self.each_example_prompts = examples.get_each_prompts()
        return self

    def distrill(self, X):
        # TODO: 未実装
        """
        Args:
            X (dict, list[dict], pd.DataFrame): json, jsons or dataframe
        Returns:
            self
        """
        return self

    def save(self, path):
        """
        モデルを保存する
        Args:
            path (str): モデルの保存先のパス
        Returns:
            self
        """
        with open(path, mode="wb") as f:
            pickle.dump(self, f)
        return self

    @recursive_apply(index=1)
    def translate_if_needed(self, text):
        """
        ユーザーに指定された場合、英訳を行う
        Args:
            text (str): 日本語テキスト
        Returns:
            text (str): 英訳済みテキスト
        """
        if self.translate_to_english:
            text = self.translator.to_english(text)
        return text