from setuptools import find_packages, setup

setup(
    name="exocortex",
    version="0.1.0",
    description="Kählerian Tensor Thermodynamics and AGI-safe computational substrate",
    author="Francisco Amador Barrios Espinoza",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=["numpy>=1.23"],
)
