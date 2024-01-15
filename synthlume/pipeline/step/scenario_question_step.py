from copy import deepcopy

from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)


class ScenarioQuestionStep(JSONStep):
    name: str = "scenario_question"

    def __init__(self, llm: LLM, language: str, retries: int = 3):
        self.prompt = Prompt.from_data(language, "question_scenario")
        super().__init__(llm, self.prompt, retries=retries)

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, dict):
            return False
        if "question" not in json_response or "answer" not in json_response:
            return False
        if not isinstance(json_response["question"], str) or not isinstance(json_response["answer"], str):
            return False

        return True

    def generate(self, inputs: dict[any]) -> dict[any]:
        output = deepcopy(inputs)

        response = super().generate(output)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None
        
        output["question"] = response["question"]
        output["answer"] = response["answer"]

        return output