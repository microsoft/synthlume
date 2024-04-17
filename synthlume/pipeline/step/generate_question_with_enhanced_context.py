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


class GenerateQuestionWithEnhancedContextStep(JSONStep):
    name: str = "enchanced_context_question"

    def __init__(
            self,
            embeddings: Embeddings,
            documents: list[Document],
            llm: LLM,
            language: str,
            n_documents: int = 3,
            min_distance: float = 0.5,
            max_distance: float = 0.95,
            retries: int = 3
            ):
        self.prompt_q = Prompt.from_data(language, "question_multicontext_generate")
        self.prompt_qd = Prompt.from_data(language, "question_multicontext_generate_d")

        super().__init__(llm, self.prompt_q, retries=retries)

        if True:
            self.vectorstore = FAISS.from_documents(documents, embeddings, distance_strategy=DistanceStrategy.COSINE)
            self.vectorstore.save_local("faiss_index")
        else:
            self.vectorstore = FAISS.load_local("faiss_index", embeddings, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
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

    def _try_get_contexts(self, query: str, n_documents, exclude_document: str = None) -> list[str]:
        embedding_vector = self.vectorstore._embed_query(query)
        most_similar = self.vectorstore.similarity_search_with_score_by_vector(embedding=embedding_vector, k=n_documents)

        most_similar = [(doc, 1 - score) for doc, score in most_similar]

        has_lower_limit = any([score < self.min_distance for _, score in most_similar])
        has_upper_limit = any([score > self.max_distance for _, score in most_similar])

        filtered = [(doc, score) for doc, score in most_similar if self.min_distance <= score <= self.max_distance]

        if exclude_document is not None:
            doc_filtered = [(doc, score) for doc, score in filtered if doc.metadata["source"] != exclude_document]
            if doc_filtered:
                filtered = doc_filtered

        if len(filtered) >= self.n_documents:
            filtered = filtered[:self.n_documents]
            return [doc.page_content for doc, _ in filtered]

        if has_lower_limit and has_upper_limit:
            return [doc.page_content for doc, _ in filtered]
        
        return self._try_get_contexts(query, 2 * n_documents)

    def _generate(self, context: str, description: str = None, current_document: str = None, custom_instruction:str="\n", **kwargs) -> dict[any]:
        contexts = self._try_get_contexts(context, self.n_documents * 2, exclude_document=current_document)

        contexts.append(context)

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