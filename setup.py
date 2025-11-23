from setuptools import setup, find_packages

setup(
    name="psifold",
    version="0.1.0",
    description="Hierarchical Recurrent Model (HRM) for RNA 2D Structure Prediction",
    author="psifold",
    py_modules=[
        "model",
        "modules",
        "training",
        "utils",
        "data",
        "evaluate",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
)
