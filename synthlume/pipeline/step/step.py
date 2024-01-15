from langchain_core.language_models.llms import LLM

from synthlume.prompts.prompt import Prompt
from synthlume.prompts.tags import Tag
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)

class Step():
    name: str = "generic"

    def __init__(
        self,
        llm: LLM,
        prompt: Prompt,
    ):
        self.llm = llm
        self.prompt = prompt


    def generate(self, inputs: dict[any]) -> any:
        logger.debug(f"Prompt keys: {', '.join(self.prompt.keys)}")

        for key in self.prompt.keys:
            assert key in inputs, f"Prompt key {key} not in inputs"

        prompt_text = self.prompt.text.format(**{key: inputs[key] for key in self.prompt.keys})

        # logger.debug(f"Prompt text:\n{prompt_text}")

        response = self.llm.invoke(prompt_text)

        if hasattr(response, "content"):
            response = response.content

        if Tag.ChainOfThought.value in self.prompt.tags:
            try:
                cot, response = response.split("-------")
            except ValueError:
                logger.debug(f"Could not split response into chain of thought and response, returning full response")
                cot = ""

            logger.debug(f"Chain of thought:\n{cot}")

        logger.debug(f"Response:\n{response}")

        return response
