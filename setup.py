import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GcomcTool",
    version="3.2.2",
    author="shoki shimada",
    author_email="shokishimada@gmail.com",
    description="This code can handle the level-2 tile products of the GCOM-C satellite datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shima-shoki/GcomcTool.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': ['sample_command = sample_command.sample_command:main']
    },
    python_requires='>=3.7',
)
