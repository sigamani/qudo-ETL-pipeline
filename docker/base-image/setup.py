from setuptools import find_packages, setup

setup(
    name="kraken",
    packages=find_packages(exclude=["kraken_tests"]),
    install_requires=[
        "dagster",
        "dagster-aws",
        "dagster-cloud",
        "dagster-webserver",
        "attrs==22.1.0",
        "boto3",
        "certifi",
        "charset-normalizer==2.1.1",
        "contourpy==1.0.6",
        "cramjam==2.6.1",
        "cycler==0.11.0",
        "exceptiongroup==1.0.1",
        "fastparquet==0.8.3",
        "fonttools==4.38.0",
        "gower==0.0.5",
        "idna==3.4",
        "imbalanced-learn==0.10.1",
        "iniconfig==1.1.1",
        "jmespath==1.0.1",
        "joblib==1.2.0",
        "kiwisolver==1.4.4",
        "kmodes==0.12.2",
        "matplotlib==3.6.2",
        "numpy==1.23.4",
        "packaging==21.3",
        "pandas==1.5.1",
        "Pillow==9.3.0",
        "pluggy==1.0.0",
        "pyarrow==10.0.1",
        "pyparsing==3.0.9",
        "pytest==7.2.0",
        "python-dateutil==2.8.2",
        "pytz==2022.6",
        "requests==2.28.1",
        "s3transfer==0.6.0",
        "s3fs",
        "scikit-learn==1.1.3",
        "scipy==1.9.3",
        "six==1.16.0",
        "sklearn==0.0",
        "threadpoolctl==3.1.0",
        "tomli==2.0.1",
        "tqdm==4.64.1",
        "urllib3==1.26.12",
        "rpy2~=3.5.6",
        "statsmodels",
        "aiobotocore==2.4.2",
        "imblearn~=0.0",
        "ray~=2.6.3",
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)


