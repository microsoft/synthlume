import random
import json

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

from synthlume.pipeline.step import (
    DescriptionStep,
    GenerateQuestionStep,
    HumanifyQuestionStep,
    ScenarioQuestionStep,
    QuestionStyleSimpleStep,
    QuestionStyleCompleteSentenseStep,
    MultipleChoiceQuestionStep,
    GenerateMulticontextQuestionStep,
    GenerateQuestionThinkingProcess,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

## constants
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-08-01-preview",
)

llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version="2023-08-01-preview",
    deployment_name=AZURE_DEPLOYMENT_NAME,
    temperature=0.9,
)


multiple_choice_step = MultipleChoiceQuestionStep(llm=llm, language="en")

with open("questions.jsonl", "r") as f:
    questions = f.readlines()
    questions = [json.loads(q) for q in questions]

modified_questions_file = open("modified_questions.jsonl", "w")

for question in questions:
    question_data = question["question"]
    response = multiple_choice_step.generate(**question_data)
    if response is None:
        print(f"Could not generate multiple choice question, skipping")
    else:
        question[multiple_choice_step.name] = response
        print(
            f"Multiple choice generated question: {question[multiple_choice_step.name]['question']}"
        )
        print(f"\tA) {question[multiple_choice_step.name]['correct_answer']}")
        print(f"\tB) {question[multiple_choice_step.name]['wrong_answer_1']}")
        print(f"\tC) {question[multiple_choice_step.name]['wrong_answer_2']}")
        print(f"\tD) {question[multiple_choice_step.name]['wrong_answer_3']}")
    print()
    print()

    modified_questions_file.write(json.dumps(question) + "\n")
