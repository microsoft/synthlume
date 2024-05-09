import os
from copy import deepcopy

from langchain_core.language_models.llms import LLM
from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)


class GenerateQuestionStep(JSONStep):
    name: str = "question"

    def __init__(
        self,
        llm: LLM,
        language: str,
        retries: int = 3,
        prompt_base: Prompt = None,
        prompt_desc: Prompt = None,
        prompt_desc_examples: Prompt = None,
    ):

        self.prompt_q = (
            prompt_base
            if prompt_base
            else self._try_load_prompt(language, "question_generate")
        )
        self.prompt_qd = (
            prompt_desc
            if prompt_desc
            else self._try_load_prompt(language, "question_generate_d")
        )
        self.prompt_qde = (
            prompt_desc_examples
            if prompt_desc_examples
            else self._try_load_prompt(language, "question_generate_de")
        )
        super().__init__(llm, self.prompt_q, retries=retries)

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, dict):
            return False
        if "question" not in json_response or "answer" not in json_response:
            return False
        if not isinstance(json_response["question"], str) or not isinstance(
            json_response["answer"], str
        ):
            return False

        return True

    def _generate(
        self, context: str, description: str = None, examples: str = None, **kwargs
    ) -> dict[any]:
        kwargs["context"] = context

        if description is not None:
            self.prompt = self.prompt_qd
            kwargs["description"] = description
        if examples is not None:
            self.prompt = self.prompt_qde
            kwargs["examples"] = examples

        assert (
            self.prompt is not None
        ), "Prompt not set. Probably it was not loaded properly"

        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")

        response = super()._generate(**kwargs)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None

        kwargs["question"] = response["question"]
        kwargs["answer"] = response["answer"]

        return kwargs
