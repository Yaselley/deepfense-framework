from setuptools import setup, find_packages

# Read requirements from requirements.txt
try:
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    name="deepfense",
    version="0.1",
    description="A Modular, Extensible Framework for Deepfake Audio Detection (ASV Spoofing)",
    packages=find_packages(),
    install_requires=requirements + ["click>=8.0.0", "omegaconf==2.0.6"],
    entry_points={
        "console_scripts": [
            "deepfense=deepfense.cli.main:cli",
        ],
    },
)
