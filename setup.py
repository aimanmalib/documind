from setuptools import setup, find_packages

setup(
    name="documind",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click>=8.1",
        "rich>=13.0",
        "httpx>=0.27",
        "pydantic>=2.5",
    ],
)
