from setuptools import setup, find_packages

setup(
    name='wire_modeling',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'sympy', 'pyarrow'
    ],
    entry_points={
        'console_scripts': [
            'run-wire-modeling = wire_modeling.run_pipeline:main'  # if you write a main()
        ]
    },
)