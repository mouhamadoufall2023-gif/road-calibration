from setuptools import setup, find_packages

setup(
    name="road-calibration",
    version="1.0.0",
    description="Calibration-aware road surface classification and braking uncertainty",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[
        # install torch and torchvision for your CUDA version first
        "torch>=1.10",
        "torchvision>=0.11",
        "hydra-core==1.1.2",          # configs rely on 1.1 behaviour (run dir = cwd)
        "opencv-python>=4.5",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "plotly",
        "terminaltables",
        "wandb",
    ],
    extras_require={"test": ["pytest"]},
)
