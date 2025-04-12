# setup.py

# Explanation of Key Components:

#     find_packages(): Automatically discovers all packages and sub-packages within your project.
#     include_package_data=True: Ensures that non-Python files (like YAML configurations) are included in the package based on the MANIFEST.in file.
#     install_requires: Lists all the dependencies your package needs. It's crucial to include all packages imported in your code here to avoid runtime errors for users.
#     classifiers: Provide metadata about your package, which helps users and tools find and understand your package better.

from setuptools import setup, find_packages
import os

# Read the contents of README.md for the long description
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Define the setup configuration
setup(
    name="datapreprocessor",  # Replace with your desired package name
    version="0.2.1",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A comprehensive data preprocessing package for machine learning models.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ghadfield32-ml_preprocessor",  # Replace with your repository URL
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files as specified in MANIFEST.in
    install_requires=[
        "fastapi",
        "uvicorn",
        "pyyaml",
        "pandas",
        "numpy",
        "scikit-learn",
        "imbalanced-learn",
        "matplotlib",
        "seaborn",
        "scipy",
        "joblib",
        "feature-engine",
        # Add any additional dependencies your package requires
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python versions you support
)
