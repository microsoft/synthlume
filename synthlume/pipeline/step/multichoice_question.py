from copy import deepcopy

from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)


class MultipleChoiceQuestionStep(JSONStep):
    name: str = "multichoice_question"
    response_keys: list[str] = [
        "wrong_answer_1",
        "wrong_answer_2",
        "wrong_answer_3",
        "correct_answer",
        "question",
    ]

    def __init__(self, llm: LLM, language: str, retries: int = 3):
        self.prompt = Prompt.from_data(language, "multichoice_question")
        super().__init__(llm, self.prompt, retries=retries)

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, dict):
            return False

        for key in self.response_keys:
            if key not in json_response:
                return False
            if not isinstance(json_response[key], str):
                return False

        return True

    def _generate(
        self,
        context: str,
        question: str,
        answer: str,
        custom_instruction: str = "\n",
        **kwargs,
    ) -> dict[any]:
        output = {
            "context": context,
            "question": question,
            "answer": answer,
            "custom_instruction": custom_instruction,
        }

        response = super()._generate(**output)

        if response is None:
            logger.warning(
                f"Could not generate multiple choice question, returning None"
            )
            return None

        for key in self.response_keys:
            output[key] = response[key]

        return output
