import json

from synthlume.metrics import (
    CosineSimilarity,
    GMMWasserstein,
    SentenceSeparabilityIndex
)
import pandas as pd
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Replace this with OpenAI embeddings as it is shown below
embeddings = SentenceTransformerEmbeddings(
    model_name="all-mpnet-base-v1"
)

base_questions = []
multichoice_questions = []
scenario_questions = []
humanify_questions = []
style_simple_questions = []
style_complete_sentence_questions = []
with open("questions.jsonl", "r") as f:
    lines = f.readlines()

    for line in lines:
        data = json.loads(line)

        base_questions.append(data["question"]["question"])
        multichoice_questions.append(data["multichoice_question"]["question"])
        scenario_questions.append(data["scenario_question"]["question"])
        humanify_questions.append(data["humanify_question"]["question"])
        style_simple_questions.append(data["style_simple"]["question"])
        style_complete_sentence_questions.append(data["style_complete_sentence"]["question"])


true_questions_set = pd.read_csv("data/moodys_data/search_evaluation.csv")["Query"].tolist()

# Use C to alternate boundaries of the metric.
# The lower the C, the more difficult it is to distinguish between the two sets due to regularization
# The higher the C, the easier it is to distinguish between the two sets
ssi = SentenceSeparabilityIndex(
    embeddings=embeddings,
    regression_kwargs={"C": 0.1},
)
gen_sentence_set = base_questions + multichoice_questions + scenario_questions + humanify_questions + style_simple_questions + style_complete_sentence_questions

# Calculate the SSI score
ssi_scores, ssi_score = ssi.evaluate_scores(true_questions_set, gen_sentence_set)

#In scores, score 1 means that it's virtually impossible to distinguish

gen_qual = zip(gen_sentence_set, list(ssi_scores))
gen_qual = sorted(gen_qual, key=lambda x: x[1], reverse=True)

for sentence, score in gen_qual:
    tag = []
    if sentence in base_questions:
        tag.append("base")
    elif sentence in multichoice_questions:
        tag.append("multichoice")
    elif sentence in scenario_questions:
        tag.append("scenario")
    elif sentence in humanify_questions:
        tag.append("humanify")
    elif sentence in style_simple_questions:
        tag.append("simple")
    elif sentence in style_complete_sentence_questions:
        tag.append("complete_sentence")

    tag = ", ".join(tag)

    print(f"Tag: {tag} - SSI Score: {score}\n\t{sentence}")

print()
print()
print(f"SSI Score: {ssi_score}")
print()
print()

ssi_top_100 = list(sorted(gen_qual, key=lambda x: x[1], reverse=True))[:100]

ssi_scores, ssi_score = ssi.evaluate_scores(true_questions_set, [item[0] for item in ssi_top_100])

print()
print()
print(f"SSI Top100 Score: {ssi_score}")
print()
print()

exit(0)

cs = CosineSimilarity(embeddings=embeddings)
gen_sentence_set = base_questions + multichoice_questions + scenario_questions + humanify_questions + style_simple_questions + style_complete_sentence_questions
# Calculate the SSI score
cs_scores, cs_score = cs.evaluate_scores(true_questions_set, gen_sentence_set)

#In scores, score 1 means that it's virtually impossible to distinguish

gen_qual = zip(gen_sentence_set, list(cs_scores))
gen_qual = sorted(gen_qual, key=lambda x: x[1], reverse=True)

for sentence, score in gen_qual:
    tag = []
    if sentence in base_questions:
        tag.append("base")
    elif sentence in multichoice_questions:
        tag.append("multichoice")
    elif sentence in scenario_questions:
        tag.append("scenario")
    elif sentence in humanify_questions:
        tag.append("humanify")
    elif sentence in style_simple_questions:
        tag.append("simple")
    elif sentence in style_complete_sentence_questions:
        tag.append("complete_sentence")

    tag = ", ".join(tag)

    print(f"Tag: {tag} - Cosine Similarity Score: {score}\n\t{sentence}")

print()
print()
print(f"Cosine Similarity Score: {cs_score}")
print()
print()