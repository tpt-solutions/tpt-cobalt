"""Setup script for tptr framework backends package."""
from setuptools import find_packages, setup

setup(
    name="tptr",
    version="0.1.0",
    description="TPT Framework Backends - Python thin wrapper over Rust GPU runtime",
    long_description="",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "pytorch": ["torch>=1.12"],
        "jax": ["jax>=0.4", "jaxlib>=0.4"],
        "dev": ["pytest>=7", "black", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software :: Libraries :: Python Modules",
    ],
)

