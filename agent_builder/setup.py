from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="agent_builder",
    version="0.1.0",
    author="MAAP Team",
    author_email="info@maapagentbuilder.io",
    description="A flexible framework for building and deploying LLM agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorganization/maap-agent-builder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agent-builder=agent_builder.cli:main",
        ],
    },
    include_package_data=True,
)
