from langchain import PromptTemplate
from llama_index import GPTSimpleVectorIndex, Document, QuestionAnswerPrompt
from .util import clean_text
from .data import Inputs, Classes, Outputs, Examples
import pandas as pd

class Classifier():
    """
    LLMを用いた分類器のクラス
    scikit-learnのインターフェースに準拠している。
    """

    def __init__(self, llm, task, classes, multi_label=False):
        """
        Args:
            llm (langchain.llms): LLMオブジェクト
            task (str): LLに指定するタスク
            classes (list[str]): 出力するラベルのリスト
            multi_label (bool): マルチラベルかどうか
        Raises:
            ValueError: data must be json, jsons or dataframe
        """
        self.llm = llm
        self.task = task
        self.classes = Classes(classes, multi_label)
        self.class_prompt = self.classes.get_prompt()
        self.multi_label = multi_label
        self.last_prompt = "A:"
        self.template = (
            "{task}\n"
            "{classes}\n"
            "{input}\n"
            "{last}"
            )
        self.prompt_template= PromptTemplate(
            input_variables=["task", "classes", "input", "last"],
            template=self.template,
        )

    def predict(self, X, return_index=True, return_df=False):
        """
        分類を行う。LLMが行ったラベル付けの結果のリストを出力する。
        ラベルはlabelsのindexで表される。
        Args:
            X (dict, list[dict], pd.DataFrame): json, jsons or dataframe
            return_index (bool): indexを返すかどうか
            return_df (bool): dataframeを返すかどうか
        Return:
            y: list[int] or list[list[int]] or list[str] or list[list[str]]
        """
        self.inputs = Inputs(X)
        input_prompts = self.inputs.get_prompts()
        self.prompts = [self.prompt_template.format(task=self.task, classes=self.class_prompt, input=input_prompt, last=self.last_prompt) for input_prompt in input_prompts]
        if hasattr(self, "llama_index"):
            responses = [self.llama_index.query(prompt, text_qa_template=self.qa_prompt) for prompt in self.prompts]
            for i, response in enumerate(responses):
                display(self.inputs.get_df().iloc[i])
                print(response.source_nodes)
            responses = [response.response for response in responses]
        else:
            responses = [self.llm(prompt) for prompt in self.prompts]
        self.outputs = Outputs.from_resposes(responses, self.classes)
        if return_index:
            y = self.outputs.get_indices()
        else:
            y = self.outputs.get_labels()
        if return_df:
            df = pd.concat([self.inputs.get_df(), pd.DataFrame({'_y': y})], axis=1)
            return df
        return y

    def fit(self, X, y, few_shot=True):
        """
        few_shot = True の場合、Xとyをプロンプトに埋め込んで予測を行う
        few_shot = False の場合、GPT-indexを用いて予測を行う
        Args:
            X (dict, list[dict], pd.DataFrame): json, jsons or dataframe
            y (list[int], list[list[int]]): ラベルのindex
            few_shot (bool): few shotかどうか
        Returns:
            self
        """
        inputs = Inputs(X)
        outputs = Outputs(y, self.classes)
        self.examples = Examples(inputs, outputs)
        if few_shot:
            self.last_prompt = self.examples.get_fewshot_prompt() + "\nA:"
        else:
            texts = self.examples.get_each_prompts()
            documents = documents = [Document(t) for t in texts]
            self.llama_index = GPTSimpleVectorIndex(documents)
            QA_PROMPT_TMPL = "{query_str}\n" + self.examples.prompt_template + "\nA:"
            self.qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)
            self.last_prompt = ""
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