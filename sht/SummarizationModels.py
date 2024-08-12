import logging
from abc import ABC, abstractmethod
import os
import time
from typing import Dict

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

import tiktoken

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, text, summary_len):
        pass

class BaseGPTSummarizationModel(BaseSummarizationModel):
    def __init__(self):

        self.model_name = None
        self.client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.user_prompt_template = "Please summarize the following text in no more than four sentences. Ensure that the summary includes all key details.\n\n[Start of the text]\n{text}\n[End of the text]"

    def summarize(self, text: str, summary_len: int) -> Dict:
        '''
        Use gpt to summarize given text.

        Args:

            - text (str): the text to be summarized

            - max_tokens (int): the limited output size

        Returns:

            A dictionary that looks like:

            - summary (str): the summary

            - input_tokens (int): input token num

            - output_tokens (int): output token num

            - time (float): time_elapsed (second)
        '''
        
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        
        if (not isinstance(summary_len, int)) or (summary_len <= 0):
            raise ValueError("summary_len must be a positive integer")
        
        logging.info(f"{self.model_name} summarizing (in {summary_len} tokens) text: {text[0:min(10, len(text))]}...")

        if len(self.tokenizer.encode(text)) <= summary_len:
            # No need for summarization
            return {
                "summary": text,
                "input_tokens": 0,
                "output_tokens": 0,
                "time": 0.0,
            }

        user_prompt = self.user_prompt_template.format(text=text)

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": user_prompt
            }],
            max_tokens=summary_len,
        )
        end_time = time.time()

        summary = response.choices[0].message.content
        input_tokens = len(self.tokenizer.encode(user_prompt))
        output_tokens = len(self.tokenizer.encode(summary))
        elapsed = end_time - start_time

        return {
            "summary": summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time": elapsed,
        }


class GPT4oMiniSummarizationModel(BaseGPTSummarizationModel):
    def __init__(self, openai_key_path: str):
        if (not isinstance(openai_key_path, str)) or (not os.path.exists(openai_key_path)):
            raise ValueError("openai_key_path should be a path to a file storing openai key")

        super().__init__()
        self.model_name = "gpt-4o-mini"
        with open(openai_key_path, 'r') as file:
            self.client = OpenAI(api_key=file.read().replace("\n", "").strip())