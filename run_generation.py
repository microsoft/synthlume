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
    GenerateMulticontextQuestionStep
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

def load_and_split(path, splitter):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return splitter.split_documents(documents)

def generate_description(documents, llm):
    description_step = DescriptionStep(llm=llm, language="en")
    text = ""
    for document in documents:
        text += document.page_content
        if len(text) > 1560:
            break
    description = description_step.generate(document=text)

    return description

def generate_questions(llm, description, documents, output_file, filename):
    questions_generatoion_step = GenerateQuestionStep(llm=llm, language="en")
    scenario_question_step = ScenarioQuestionStep(llm=llm, language="en")
    humanify_question_step = HumanifyQuestionStep(llm=llm, language="en")
    question_style_simple_step = QuestionStyleSimpleStep(llm=llm, language="en")
    complete_sentence_step = QuestionStyleCompleteSentenseStep(llm=llm, language="en")
    multiple_choice_step = MultipleChoiceQuestionStep(llm=llm, language="en")

    results = []

    random.shuffle(documents)
    max_steps = 5
    steps_done = 0

    for i, chunk in enumerate(documents):
        if steps_done >= max_steps:
            break
        try:
            metadata = chunk.metadata
            chunk = chunk.page_content
            print(f"Chunk {steps_done+1}/{max_steps}")
            calls = {}

            inputs = {
                "context": chunk,
                "description": description,
                "current_document": metadata["source"]
            }

            calls["input"] = inputs

            response = multicontext_generation_step.generate(**inputs)

            # response = questions_generatoion_step.generate(**inputs)

            if response is None:
                print(f"Could not generate question, skipping")
                continue

            calls[questions_generatoion_step.name] = response
            print(f"Base generated question: {calls[questions_generatoion_step.name]['question']}")
            print(f"\tAnswer: {calls[questions_generatoion_step.name]['answer']}")
            print()
            print()

            response = multiple_choice_step.generate(**calls[questions_generatoion_step.name])
            if response is None:
                print(f"Could not generate multiple choice question, skipping")
            else:
                calls[multiple_choice_step.name] = response
                print(f"Multiple choice generated question: {calls[multiple_choice_step.name]['question']}")
                print(f"\tA) {calls[multiple_choice_step.name]['correct_answer']}")
                print(f"\tB) {calls[multiple_choice_step.name]['wrong_answer_1']}")
                print(f"\tC) {calls[multiple_choice_step.name]['wrong_answer_2']}")
                print(f"\tD) {calls[multiple_choice_step.name]['wrong_answer_3']}")

            print()
            print()
            print("     --------     ")
            print()
            print()

            calls["filename"] = filename

            output_file.write(json.dumps(calls) + "\n")

            results.append(calls)
            steps_done += 1
        except Exception as e:
            print("########## ERROR ##########")
            print(f"Error: {e}")
            print("###########################")
    
    return results


## constants
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")

base_path = "data/moodys_data"
pdfs = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if filename.endswith(".pdf")]

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2048,
    chunk_overlap=256,
    length_function=len,
    is_separator_regex=False,
)

all_documents = sum([load_and_split(pdf, text_splitter) for pdf in pdfs], [])

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


multicontext_generation_step = GenerateMulticontextQuestionStep(
    llm=llm,
    language="en",
    documents=all_documents,
    embeddings=embeddings,
    n_documents=10,
    min_distance=0.85,
    max_distance=0.98,
)

for pdf in pdfs[1:]:
    print(f"Processing {pdf}")
    documents = load_and_split(pdf, text_splitter)
    description = generate_description(documents, llm)
    with open("questions.jsonl", "a") as output_file:
        results = generate_questions(llm, description, documents, output_file, pdf)
        print(f"Generated {len(results)} questions for {pdf}")
