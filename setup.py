from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="daria",
    version="0.0.1",
    description="Darcy scale image analysis toolbox",
    keywords="darcy image analysis porous media flow",
    # py_modules={"helloworld"},
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"daria": ["py.typed"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python ::3",
        "Programming Language :: Python ::3.10"
        "License :: OSI Approved :: Apache v2 License",
        "Operating System :: Linux, Windows",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "opencv-python",
    ],
    extras_require={
        "dev": [
            "pytest>=7.1",
            "black == 22.3.0",
        ],
    },
    python_requires=">=3",
    url="https://github.com/EStorvik/DarIA.git",
    author="Erlend Storvik, Jan Nordbotten and Jakub Wiktor Both",
    maintainer="Erlend Storvik",
    maintainer_email="erlend.storvik@uib.no",
    platforms=["Linux", "Windows"],
    license="Apache v2",
)
