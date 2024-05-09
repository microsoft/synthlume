from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="synthlume",
    version="0.1",
    packages=find_packages(),
    description="Synthetic Q&A dataset generator for NLP tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vadim Kirilin",
    author_email="vadimkirilin@microsoft.com",
    url="https://github.com/quovadim/synthlume",
    install_requires=required,
    classifiers=[],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "synthlume": ["prompts/en/*.txt"],
    },
)
