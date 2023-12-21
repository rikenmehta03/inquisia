from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="inquisia",
    version="0.1.0",
    description="Inquisia: Knowledge Querying System using LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rikenmehta03/inquisia",
    author="Riken Mehta",
    author_email="riken.mehta03@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="llm, rag, chat, productivity, gpt, langchain, ai",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
)
