from setuptools import find_packages, setup


setup(
    name="rtcore",
    version="0.1.0",
    description="Deterministic tensor and mesh processing utilities",
    author="Francisco Amador Barrios Espinoza",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "numpy",
        "trimesh",
        "opencv-python",
    ],
)
