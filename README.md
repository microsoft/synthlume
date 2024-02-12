# SynthLume

Synthlume is an advanced tool designed for generating artificial Q&A scenarios, facilitating the testing and development of Q&A systems. It streamlines the creation of realistic questions and answers, enhancing the evaluation process for these applications.

## Simplified Q&A Generation
Synthlume streamlines the creation of questions from specified contexts. Each question is crafted to be relevant, answerable, and understandable on its own, ensuring a high degree of relevance and coherence.

Diverse Question Formulation

The platform introduces variability in question formulation, enabling users to explore different ways of asking the same question. For instance, a query about the impact of artificial intelligence on personalized medicine could be rephrased in multiple ways to explore various aspects and implications, enriching the dataset with diverse perspectives.

## Evaluation Metrics

To assess the quality and realism of generated questions, Synthlume employs two primary metrics: Cosine Similarity and the Sentence Separability Index (SSI).

 - Cosine Similarity: This metric measures the similarity between generated and actual questions based on their representation in a multidimensional space, offering insights into the variance and accuracy of the generated content.
 - Sentence Separability Index (SSI): Inspired by Generative Adversarial Networks, SSI differentiates between real and synthetic questions by training a model to recognize subtle differences, allowing for refined adjustment based on project needs.

## Enhancing Multiple Choice Questions

Synthlume excels in creating multiple-choice questions, a critical component in evaluating Q&A systems. This feature tests both the system's ability to retrieve relevant information and its capacity to discern the correct answer from plausible alternatives. By generating both correct and intentionally incorrect options, Synthlume provides a comprehensive framework for robust system evaluation.

# Getting Started

Before you are able to generate text, you need to prepare your enviroment.

You can either use Virtual Enviorment of Conda.


## Option 1: Using Virtual Env

### Create your Virtual Environment

We suggest you create an enviroment named venv_lume:
```bash
python -m venv ./venv_lume
```

### Activate your enviroment

To activate your enviroment, use one of the commands below:

| Platform | Shell   | Command to activate virtual environment        |
|----------|---------|------------------------------------------------|
| POSIX    | bash/zsh| $ source ./venv_lume/bin/activate                   |
|          | fish    | $ source ./venv_lume/bin/activate.fish              |
|          | csh/tcsh| $ source ./venv_lume/bin/activate.csh               |
| PowerShell|         | $ ./venv_lume/bin/Activate.ps1                 |
| Windows  | cmd.exe | C:\> venv_lume\Scripts\activate.bat               |
|          | PowerShell | PS C:\> venv_lume\Scripts\Activate.ps1          |

### Install the Requirements
```
pip install -e .
```

## Option 2: Using Conda

If you prefer to use conda, use the following commands to create a conda enviroment named sys

```bash
conda create -n lume python==3.10
conda activate
conda activate lume
pip install -e .
```

### Create your environment variables.

Copy file **.env-template** onto **.env**

Update the keys as shown below:

```
AZURE_OPENAI_KEY=<your openain key>
AZURE_ENDPOINT=<your open ai endpoint>
AZURE_DEPLOYMENT_NAME=<the name of your gpt-4 deployment>
```
## Running experiment

### Option 1: Python Script

Once your python enviroment has been activated, run:

```
python -m run_generation.py
```

### Option 2: Jupyter notebook

run notebook [notebooks/generation.ipynb](notebooks/generation.ipynb)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
