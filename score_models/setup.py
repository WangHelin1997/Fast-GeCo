from setuptools import setup, find_packages

setup(
	name="score_models",
	version="0.5.11",
    description="A simple pytorch interface for score model and basic diffusion.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "scipy",
        "torch_ema",
        "h5py",
        "numpy",
        "tqdm"
    ],
	python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)

