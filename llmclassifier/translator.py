class LLMTranslator():
    """翻訳機のクラス
    """
    def __init__(self, llm):
        self.llm = llm

    def translate(self, text, src_lang, dest_lang):
        """翻訳を行う
        Args:
            text (str): 翻訳するテキスト
            src_lang (str): 入力言語
            dest_lang (str): 出力言語
        Return:
            translated_text (str): 翻訳されたテキスト
        """
        prompt = f"""Translate from {src_lang} to {dest_lang}. 
        Do not output anything other than the translated text. 
        Especially, never leave the original language.
        Translate the following sentence : {text} 
        A: """
        translated_text = self.llm(prompt)
        return translated_text

    def to_english(self, text, src_lang="Japanese"):
        """英語に翻訳を行う
        Args:
            text (str): 翻訳するテキスト
            src_lang (str): 入力言語
        Return:
            translated_text (str): 翻訳されたテキスト
        """
        return self.translate(text, src_lang, "English")

    def from_english(self, text, dest_lang="Japanese"):
        """英語から翻訳を行う
        Args:
            text (str): 翻訳するテキスト
            dest_lang (str): 出力言語
        Return:
            translated_text (str): 翻訳されたテキスト
        """
        return self.translate(text, "English", dest_lang)