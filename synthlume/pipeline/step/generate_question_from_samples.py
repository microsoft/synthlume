import os
from copy import deepcopy

from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document

from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)


class GenerateQuestionFromSamples(JSONStep):
    name: str = "question"

    def __init__(
        self,
        llm: LLM,
        language: str,
        retries: int = 3,
        prompt: Prompt = None,
    ):

        self.prompt = (
            prompt
            if prompt
            else self._try_load_prompt(language, "questions_from_samples")
        )
        super().__init__(llm, self.prompt, retries=retries)

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, list):
            return False
        if not all(isinstance(item, dict) for item in json_response):
            return False
        if not all("question" in item or "answer" in item for item in json_response):
            return False
        if not all(isinstance(item["question"], str) or isinstance(item["answer"], str) for item in json_response):
            return False
        return True

    def _generate(
        self, variants: list[str], context: list[Document], **kwargs
    ) -> dict[any]:
        kwargs["context"] = "\n\n".join(item.page_content for item in context)

        assert (
            self.prompt is not None
        ), "Prompt not set. Probably it was not loaded properly"

        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")
        
        kwargs["variants"] = "\n\n".join(variants)

        response = super()._generate(**kwargs)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None

        kwargs["questions"] = response
        kwargs["context"] = context

        return kwargs
