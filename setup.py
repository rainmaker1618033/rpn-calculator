# ============================================================================
# FILE: setup.py (optional, for installation)
# ============================================================================
"""
from setuptools import setup, find_packages

setup(
    name="rpn_calculator",
    version="1.0.0",
    description="A modular scientific RPN calculator",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",  # Optional, for LU, SCHUR, HESSENBERG
    ],
    entry_points={
        'console_scripts': [
            'rpn-calc=rpn_calculator.cli:main',
        ],
    },
    python_requires='>=3.8',
)
"""