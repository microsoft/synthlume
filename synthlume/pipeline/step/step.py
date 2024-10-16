from langchain_core.language_models.llms import LLM

from synthlume.prompts.prompt import Prompt
from synthlume.prompts.tags import Tag
from synthlume.pipeline.core.runnable import Runnable
from synthlume.logging.logging import get_logger

logger = get_logger(__name__)


class Step(Runnable):
    name: str = "generic"

    def __init__(
        self,
        llm: LLM,
        prompt: Prompt,
    ):
        self.llm = llm
        self.prompt = prompt
        super().__init__()

    @staticmethod
    def _try_load_prompt(language: str, name: str) -> Prompt:
        try:
            prompt = Prompt.from_data(language, name)
        except FileNotFoundError:
            logger.warning(
                f"Could not find prompt for language {language}, using default"
            )
            prompt = None

        return prompt

    def _generate(self, **kwargs) -> any:
        logger.debug(f"Prompt keys: {', '.join(self.prompt.keys)}")

        for key in self.prompt.keys:
            assert key in kwargs, f"Prompt key {key} not in inputs"
        if "custom_instruction" not in self.prompt.keys:
            kwargs["custom_instruction"] = "\n"
        else:
            kwargs["custom_instruction"] = (
                f"\nThis is the most important instruction for you. Check that your response is with compliance with it"
                f"\n{kwargs['custom_instruction']}\n"
            )

        prompt_text = self.prompt.text.format(
            **{key: kwargs[key] for key in self.prompt.keys}
        )

        logger.debug(f"Prompt text:\n{prompt_text}")

        response = self.llm.invoke(prompt_text)

        logger.debug(f"Response:\n{response}")

        if hasattr(response, "content"):
            response = response.content

        if Tag.ChainOfThought.value in self.prompt.tags:
            try:
                cot, response = response.split("##RESPONSE##")
                logger.debug(f"CoT:\n{cot}")
            except ValueError:
                logger.debug(
                    f"Could not split response into chain of thought and response, returning full response"
                )
                cot = ""

            logger.debug(f"Chain of thought:\n{cot}")

        logger.debug(f"Response:\n{response}")

        return response
