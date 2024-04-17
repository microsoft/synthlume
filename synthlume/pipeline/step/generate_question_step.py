import os
from copy import deepcopy

from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)


class GenerateQuestionStep(JSONStep):
    name: str = "question"

    def __init__(self, llm: LLM, language: str, retries: int = 3):
        self.prompt_q = Prompt.from_data(language, "question_generate")
        self.prompt_qd = Prompt.from_data(language, "question_generate_d")
        self.prompt_qde = Prompt.from_data(language, "question_generate_de")
        super().__init__(llm, self.prompt_q, retries=retries)

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, dict):
            return False
        if "question" not in json_response or "answer" not in json_response:
            return False
        if not isinstance(json_response["question"], str) or not isinstance(json_response["answer"], str):
            return False

        return True

    def _generate(self, context: str, description: str = None, examples: str = None, custom_instruction:str="\n", **kwargs) -> dict[any]:
        output = {
            "context": context,
            "custom_instruction": custom_instruction
        }

        if description is not None:
            self.prompt = self.prompt_qd
            output["description"] = description
        if examples is not None:
            self.prompt = self.prompt_qde
            output["examples"] = examples

        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")

        response = super()._generate(**output)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None
        
        output["question"] = response["question"]
        output["answer"] = response["answer"]

        return output