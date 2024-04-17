import os
from copy import deepcopy

from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy

from synthlume.pipeline.step.json_step import JSONStep
from synthlume.prompts.prompt import Prompt
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)


class GenerateMulticontextQuestionStep(JSONStep):
    name: str = "multicontext_question"

    def __init__(
            self,
            llm: LLM,
            language: str,
            retries: int = 3
            ):
        self.prompt_q = Prompt.from_data(language, "question_multicontext_generate")
        self.prompt_qd = Prompt.from_data(language, "question_multicontext_generate_d")

        super().__init__(llm, self.prompt_q, retries=retries)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.n_documents = n_documents

    def validate(self, json_response: any) -> bool:
        if not isinstance(json_response, dict):
            return False
        if "question" not in json_response or "answer" not in json_response:
            return False
        if not isinstance(json_response["question"], str) or not isinstance(json_response["answer"], str):
            return False

        return True

    def _generate(self, contexts: list[str], description: str = None, custom_instruction:str="\n", **kwargs) -> dict[any]:
        merged_context = [f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)]
        merged_context = "\n\n".join(merged_context)

        output = {
            "context": merged_context,
            "custom_instruction": custom_instruction
        }

        if description is not None:
            self.prompt = self.prompt_qd
            output["description"] = description

        logger.debug(f"Using prompt {os.path.basename(self.prompt.path)}")

        response = super()._generate(**output)

        if response is None:
            logger.warning(f"Could not generate question, returning None")
            return None
        
        output["question"] = response["question"]
        output["answer"] = response["answer"]
        output["raw_contexts"] = contexts

        return output