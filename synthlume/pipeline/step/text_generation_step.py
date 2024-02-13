import os

from copy import deepcopy
from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.step import Step
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)

class TextGenerationStep(Step):
    output_key: str = "text_gen"
    prompt_name: str = "text_gen"

    def __init__(self, llm: LLM, language: str):
        prompt = Prompt.from_data(language, self.prompt_name)
        super().__init__(llm, prompt)

    def _generate(self, **kwargs) -> dict[any]:
        output = deepcopy(kwargs)
        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")

        response = super()._generate(**output)
        output[self.output_key] = response

        return output

class HumanifyQuestionStep(TextGenerationStep):
    name: str = "humanify_question"

    output_key: str = "question"
    prompt_name: str = "question_humanify"

    def _generate(self, question: str, **kwargs) -> dict:
        return super()._generate(question=question)

class DescriptionStep(TextGenerationStep):
    name: str = "description"

    output_key: str = "description"
    prompt_name: str = "description"

    def _generate(self, document: str, **kwargs) -> dict:
        return super()._generate(document=document)

class QuestionStyleSimpleStep(TextGenerationStep):
    name: str = "style_simple"

    output_key: str = "question"
    prompt_name: str = "question_style_simple"

    def _generate(self, question: str, **kwargs) -> dict:
        return super()._generate(question=question)

class QuestionStyleCompleteSentenseStep(TextGenerationStep):
    name: str = "style_complete_sentence"

    output_key: str = "question"
    prompt_name: str = "question_style_complete_sentence"

    def _generate(self, question: str, **kwargs) -> dict:
        return super()._generate(question=question)
    
class QuestionNewStyleStep(TextGenerationStep):
    name: str = "new_style"

    output_key: str = "question"
    prompt_name: str = "question_new_style"

    def _generate(self, question: str, new_style: str, **kwargs) -> dict:
        return super()._generate(question=question, new_style=new_style)
