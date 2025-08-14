from setuptools import setup, find_packages
import os

# Read requirements from the requirements.txt file
with open(os.path.join("agent_builder", "requirements.txt"), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="agent_builder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
