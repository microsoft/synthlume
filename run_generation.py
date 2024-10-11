import random
import copy

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
    GenerateQuestionWithEnhancedContextStep,
    GenerateQuestionThinkingProcess,
    GenerateQuestionFromSamples,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

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
    generate_from_samples_step = GenerateQuestionFromSamples(llm=llm, language="en")

    results = []

    random.shuffle(documents)
    max_steps = 5
    steps_done = 0

    for i, chunk in enumerate(documents):
        if steps_done >= max_steps:
            break
        try:
            metadata = chunk.metadata
            chunk = chunk
            print(f"Chunk {steps_done+1}/{max_steps}")
            calls = {}

            inputs = {
                "context": chunk,
                "description": description,
                "current_document": metadata["source"],
            }

            calls["input"] = inputs
            
            samples_response = sampling_generation.generate(**inputs)
            
            generation = generate_from_samples_step.generate(**samples_response)
            
            for option in generation["questions"]:
                print()
                print(f"Question: {option['question']}")
                print(f"Answer: {option['answer']}")
                print()
                
            flatten = []
            for option in generation["questions"]:
                item = copy.deepcopy(generation)
                del item["questions"]
                item["question"] = option["question"]
                item["answer"] = option["answer"]
                flatten.append(item)
                
        except Exception as e:
            print("########## ERROR ##########")
            print(f"Error: {e}")
            print("###########################")

    return results


## constants
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

base_path = "data/moodys_data"
pdfs = [
    os.path.join(base_path, filename)
    for filename in os.listdir(base_path)
    if filename.endswith(".pdf")
]

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
    openai_api_version="2024-09-01-preview",
)

llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version="2024-09-01-preview",
    deployment_name=AZURE_DEPLOYMENT_NAME,
    temperature=0.9,
)

llm_small = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version="2024-09-01-preview",
    deployment_name="gpt-4o-mini",
    temperature=0.9,
)

sampling_generation = GenerateQuestionThinkingProcess(
    llm=llm_small,
    language="en",
    documents=all_documents,
    embeddings=embeddings,
    n_samples=15,
    n_documents=10,
    min_distance=0.85,
    max_distance=0.98,
)


multicontext_generation_step = GenerateQuestionWithEnhancedContextStep(
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
    description = None#generate_description(documents, llm)
    with open("questions.jsonl", "a") as output_file:
        results = generate_questions(llm, description, documents, output_file, pdf)
        print(f"Generated {len(results)} questions for {pdf}")
