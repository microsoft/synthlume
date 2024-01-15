import json

from abc import ABC, abstractmethod
from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.step import Step
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)


class JSONStep(Step, ABC):
    name: str = "generic_json"

    def __init__(
            self,
            llm: LLM,
            prompt: Prompt,
            retries: int = 3
        ):
        super().__init__(llm, prompt)
        self.retries = retries

    @abstractmethod
    def validate(self, json_response: any) -> bool:
        pass

    def generate(self, inputs: dict[any]) -> any:
        json_response = None

        for r in range(self.retries):
            response = super().generate(inputs)

            try:
                json_response = json.loads(response)
            except Exception as e:
                logger.warning(f"Could not decode response as JSON, retrying ({r+1}/{self.retries})")
                logger.warning(f"Error: {e}")
                continue

            if self.validate(json_response):
                break
            else:
                logger.warning(f"Response did not pass validation, retrying ({r+1}/{self.retries})")

        return json_response
        


