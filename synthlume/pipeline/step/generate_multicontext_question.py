import os
from copy import deepcopy

from langchain_core.language_models.llms import LLM

from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from langchain_core.documents import Document
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)


class GenerateMulticontextQuestionStep(JSONStep):
    name: str = "multicontext_question"

    def __init__(
        self,
        llm: LLM,
        language: str,
        retries: int = 3,
        prompt_base: Prompt = None,
        prompt_desc: Prompt = None,
    ):
        self.prompt_q = (
            prompt_base
            if prompt_base
            else self._try_load_prompt(language, "question_multicontext_generate")
        )
        self.prompt_qd = (
            prompt_desc
            if prompt_desc
            else self._try_load_prompt(language, "question_multicontext_generate_d")
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
        self,
        contexts: list[Document],
        description: str = None,
        custom_instruction: str = "\n",
        **kwargs,
    ) -> dict[any]:
        merged_context = [
            f"Context {i+1}:\n{c.page_content}" for i, c in enumerate(contexts)
        ]
        merged_context = "\n\n".join(merged_context)

        output = {"context": merged_context, "custom_instruction": custom_instruction}

        if description is not None:
            self.prompt = self.prompt_qd
            output["description"] = description

        assert (
            self.prompt is not None
        ), "Prompt not set. Probably it was not loaded properly"

        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")

        response = super()._generate(**output)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None

        output["question"] = response["question"]
        output["answer"] = response["answer"]
        output["context"] = contexts

        return output
