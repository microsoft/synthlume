from synthlume.metrics import (
    CosineSimilarity,
    GMMWasserstein,
    SentenceSeparabilityIndex
)
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Replace this with OpenAI embeddings as it is shown below
embeddings = SentenceTransformerEmbeddings(
    model_name="all-mpnet-base-v1"
)

# from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
# embeddings = AzureOpenAIEmbeddings(
#     openai_api_key="",
#     azure_endpoint="",
#     openai_api_version="2023-08-01-preview",
#     deployment_name="",
# )

ssi = SentenceSeparabilityIndex(
    embeddings=embeddings,
)

true_sentence_set = [
    "What inspired the development of the first computer programming language?",
    "How does quantum computing differ from classical computing?",
    "What are the primary ethical considerations in AI development?",
    "What's the future of blockchain in financial transactions?",
    "How has machine learning impacted medical diagnostics?",
    "What role do neural networks play in natural language processing?",
    "What are the limitations of current virtual reality technology?",
    "How do self-driving cars handle complex traffic scenarios?",
    "What advancements are being made in cyber security?",
    "How is big data influencing personalized marketing strategies?",
]

gen_sentence_set_alternative = [
    # These are normal generated questions
    "What are the main challenges in scaling a database for large applications?",
    "How can edge computing enhance the Internet of Things (IoT)?",
    "What strategies are effective for preventing data breaches in cloud computing?",
    "How does reinforcement learning differ from supervised learning in AI?",
    "What are the latest advancements in 3D printing technology?",
    "How are biometrics being used to enhance security systems?",
    "What are the ethical implications of deepfake technology?",
    "How is artificial intelligence being integrated into renewable energy systems?",
    "What are the key components of effective DevOps practices?",
    "How is augmented reality changing the landscape of mobile app development?",
]
gen_sentence_set_reformulations = [
    # These are true questions but reformulated
    "What led to the creation of the earliest computer programming language?",
    "In what ways does quantum computing stand apart from traditional computing?",
    "What key ethical issues arise in the development of artificial intelligence?",
    "What's the potential role of blockchain in future financial transactions?",
    "How is machine learning revolutionizing the field of medical diagnostics?",
    "What significance do neural networks hold in processing natural language?",
    "What are some current limitations faced by virtual reality technologies?",
    "How do autonomous vehicles manage in complex traffic situations?",
    "What are recent developments in the field of cybersecurity?",
    "In what ways is big data reshaping personalized marketing tactics?",
]
gen_sentence_set_redneck = [
    # These are random redneck style questions with lots of mistakes
    "How come them computers ain't gettin' no viruses like us folks do?",
    "Ain't quantum computin' just some kinda space magic or sumthin'?",
    "Why we botherin' with AI when we got good ol' horse sense?",
    "Them blockchain thingamajigs, they like fancy piggy banks or what?",
    "Why's them smart learnin' machines needin' to poke around in doctorin'?",
    "Ain't them neural netwhatchamacallits just a buncha tangled fishin' lines?",
    "Them virtual whatzits, they just fancy goggles for play pretend, right?",
    "How them self-steerin' wagons 'posed to know 'bout stoppin' at mud holes?",
    "Ain't no computer smarty-pants gonna outwit a good coon dog in security, right?",
    "Big data sounds like a big ol' pile of cow dung. Why's it matter to them city slickers?",
]
gen_sentence_set_original = [
    # These are true sentence set
    "What inspired the development of the first computer programming language?",
    "How does quantum computing differ from classical computing?",
    "What are the primary ethical considerations in AI development?",
    "What's the future of blockchain in financial transactions?",
    "How has machine learning impacted medical diagnostics?",
    "What role do neural networks play in natural language processing?",
    "What are the limitations of current virtual reality technology?",
    "How do self-driving cars handle complex traffic scenarios?",
    "What advancements are being made in cyber security?",
    "How is big data influencing personalized marketing strategies?",
]
gen_sentence_set_random = [
    "Banana, highway, telescope, laughter.",
    "Umbrella, dragonfly, notebook, ocean.",
    "Symphony, pineapple, bridge, flashlight.",
    "Glitter, kangaroo, snowflake, library.",
    "Rainbow, tractor, lighthouse, melody.",
    "Cupcake, astronaut, forest, kite.",
    "Dolphin, clock, mountain, typewriter.",
    "Butterfly, guitar, volcano, pillow.",
    "Marshmallow, telescope, river, skateboard.",
    "Sunflower, igloo, thunderstorm, chess.",
]

gen_sentence_set = gen_sentence_set_reformulations + gen_sentence_set_alternative + gen_sentence_set_original + gen_sentence_set_redneck + gen_sentence_set_random
# Calculate the SSI score
ssi_scores = ssi.evaluate_scores(true_sentence_set, gen_sentence_set)

#In scores, score 1 means that it's virtually impossible to distinguish

gen_qual = zip(gen_sentence_set, list(ssi_scores))
gen_qual = sorted(gen_qual, key=lambda x: x[1], reverse=True)

for sentence, score in gen_qual:
    tag = ""
    if sentence in gen_sentence_set_alternative:
        tag = "alternative"
    elif sentence in gen_sentence_set_reformulations:
        tag = "reformulation"
    elif sentence in gen_sentence_set_redneck:
        tag = "redneck"
    elif sentence in gen_sentence_set_original:
        tag = "original"
    elif sentence in gen_sentence_set_random:
        tag = "random"

    print(f"Tag: {tag} - SSI Score: {score}\n\t{sentence}")
