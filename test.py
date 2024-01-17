import json
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.chat_models import AzureChatOpenAI

from synthlume.metrics.ssi import SentenceSeparabilityIndex
from synthlume.metrics.cosine_similarity import CosineSimilarity
from synthlume.metrics.gmm_wasserstain import GMMWasserstein
from synthlume.pipeline.step import (
    DescriptionStep,
    GenerateQuestionStep,
    HumanifyQuestionStep,
    ScenarioQuestionStep,
    QuestionStyleSimpleStep,
    QuestionStyleCompleteSentenseStep,
    MultipleChoiceQuestionStep
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

with open(f"data/sample_texts.txt", "r") as f:
    data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1024,
    chunk_overlap=256,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([data])

llm = AzureChatOpenAI(
    openai_api_key="7fbb2519691c4720b613b409f38507fa",
    azure_endpoint="https://synthlume-vadim.openai.azure.com/",
    openai_api_version="2023-08-01-preview",
    deployment_name="gpt-4",
    temperature=0.9,
)

description_step = DescriptionStep(llm=llm, language="en")

description = description_step.generate(document=data[:2048])
description = description[description_step.output_key]

"""questions_generatoion_step = GenerateQuestionStep(llm=llm, language="en")
multiple_choice_step = MultipleChoiceQuestionStep(llm=llm, language="en")
humanify_question_step = HumanifyQuestionStep(llm=llm, language="en")
scenario_question_step = ScenarioQuestionStep(llm=llm, language="en")

pipe = questions_generatoion_step | (multiple_choice_step & humanify_question_step & scenario_question_step)

res = pipe.generate(
    context=texts[0].page_content,
    description=description,
)

print(res)

exit(0)"""

questions_generatoion_step = GenerateQuestionStep(llm=llm, language="en")
scenario_question_step = ScenarioQuestionStep(llm=llm, language="en")
humanify_question_step = HumanifyQuestionStep(llm=llm, language="en")
question_style_simple_step = QuestionStyleSimpleStep(llm=llm, language="en")
complete_sentence_step = QuestionStyleCompleteSentenseStep(llm=llm, language="en")
multiple_choice_step = MultipleChoiceQuestionStep(llm=llm, language="en")

results = []

output_jsonl = open("data/output.jsonl", "w")

for i, chunk in enumerate(texts[:2]):
    chunk = chunk.page_content
    print(f"Chunk {i+1}/{len(texts)}")
    calls = {}

    inputs = {
        "context": chunk,
        "description": description
    }

    calls["input"] = inputs

    response = questions_generatoion_step.generate(**inputs)

    if response is None:
        print(f"Could not generate question, skipping")
        continue

    calls[questions_generatoion_step.name] = response
    print(f"Base generated question: {calls[questions_generatoion_step.name]['question']}")

    response = multiple_choice_step.generate(**calls[questions_generatoion_step.name])
    if response is None:
        print(f"Could not generate multiple choice question, skipping")
    else:
        calls[multiple_choice_step.name] = response
        print(f"Multiple choice generated question: {calls[multiple_choice_step.name]['question']}")
        print(f"\tA) {calls[multiple_choice_step.name]['answer']}")
        print(f"\tB) {calls[multiple_choice_step.name]['wrong_answer_1']}")
        print(f"\tC) {calls[multiple_choice_step.name]['wrong_answer_2']}")
        print(f"\tD) {calls[multiple_choice_step.name]['wrong_answer_3']}")

    response = scenario_question_step.generate(**calls[questions_generatoion_step.name])
    if response is None:
        print(f"Could not generate scenario question, skipping")
    else:
        calls[scenario_question_step.name] = response
        print(f"Scenario generated question: {calls[scenario_question_step.name]['question']}")

    response = humanify_question_step.generate(**calls[questions_generatoion_step.name])
    if response is None:
        print(f"Could not generate human-like question, skipping")
    else:
        calls[humanify_question_step.name] = response
        print(f"Human-like generated question: {calls[humanify_question_step.name]['question']}")

    response = question_style_simple_step.generate(**calls[questions_generatoion_step.name])
    if response is None:
        print(f"Could not generate simple question, skipping")
    else:
        calls[question_style_simple_step.name] = response
        print(f"Simple generated question: {calls[question_style_simple_step.name]['question']}")

    response = complete_sentence_step.generate(**calls[questions_generatoion_step.name])
    if response is None:
        print(f"Could not generate complete sentence question, skipping")
    else:
        calls[complete_sentence_step.name] = response
        print(f"Complete sentence generated question: {calls[complete_sentence_step.name]['question']}")

    print()

    output_jsonl.write(json.dumps(calls) + "\n")

    results.append(calls)

data = []
for call in results:
    row = {}
    row["context"] = call["input"]["context"]
    for name in [
        questions_generatoion_step.name,
        scenario_question_step.name,
        humanify_question_step.name,
        question_style_simple_step.name,
        complete_sentence_step.name,
    ]:
        if name not in call:
            continue
        row[f"{name}_question"] = call[name]["question"]
        row[f"{name}_answer"] = call[name]["answer"]

    if multiple_choice_step.name not in call:
        continue
    row[f"{multiple_choice_step.name}_question"] = call[multiple_choice_step.name]["question"]
    row[f"{multiple_choice_step.name}_answer"] = call[multiple_choice_step.name]["answer"]
    row[f"{multiple_choice_step.name}_wrong_answer_1"] = call[multiple_choice_step.name]["wrong_answer_1"]
    row[f"{multiple_choice_step.name}_wrong_answer_2"] = call[multiple_choice_step.name]["wrong_answer_2"]
    row[f"{multiple_choice_step.name}_wrong_answer_3"] = call[multiple_choice_step.name]["wrong_answer_3"]

    data.append(row)

df = pd.DataFrame(data)
df.to_csv("data/questions.csv", index=False)

with open("data/questions.json", "w") as f:
    json.dump(results, f, indent=4)

