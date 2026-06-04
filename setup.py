from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pfmu",
    version="1.0.0",
    author="V. Venkataramanan, Pankaj Mishra, Aarit Mehta, Jaychand Upadhyay, Supriya Dicholkar, Vats Shah, Aditya Ravi",
    author_email="pankaj.mishra@somaiya.edu",
    description="Fog-Based Unified Mobility Framework for Coordinated Traffic Signal Control and Smart Parking Allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/pfmu",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pfmu-run=experiments.run_pfmu:main",
            "pfmu-baselines=experiments.run_baselines:main",
        ],
    },
)
