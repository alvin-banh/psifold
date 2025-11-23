from setuptools import setup, find_packages

setup(
    name="psifold",
    version="0.1.0",
    description="Hierarchical Recurrent Model (HRM) for RNA 2D Structure Prediction",
    author="psifold",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
    ],
    python_requires=">=3.8",
)
