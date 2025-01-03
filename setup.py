from setuptools import setup, find_packages

setup(
    name="datapreprocessor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "imblearn",
        # ... more if needed
    ],
    author="YourName",
    author_email="your_email@example.com",
    description="A preprocessor package for ML tasks with optional clustering.",
    url="https://github.com/yourusername/yourrepo",
)
