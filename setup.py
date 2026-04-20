from setuptools import setup, find_packages

setup(
    name="pigollum",
    version="0.1.0",
    description="Principle-Guided Bayesian Optimization (PiGollum) for scientific discovery",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "botorch",
        "openai",
        "sentence-transformers",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
    ],
)
