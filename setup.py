from setuptools import find_packages, setup

install_requires = ["ogb", "hydra-core"]
setup_requires = ["pytest-runner"]
tests_require = ["pytest", "pytest-cov"]

setup(
    name="microGNN",
    version="0.0.1",
    author="Juyi Lin",
    author_email="juyi.lin@kaust.edu.sa",
    description="divide minibatch to nano batch",
    python_requires=">=3.6",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={"test": tests_require},
    packages=find_packages(),
)
