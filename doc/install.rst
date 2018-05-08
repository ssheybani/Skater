Install Skater
================

Dependencies
~~~~~~~~~~~~~~~~
Skater relies on 

- scikit-learn>=0.18,
- pandas>=0.19,
- ds-lime>=0.1.1.21(datascience.com forked version of LIME),
- requests,
- multiprocess,
- joblib==0.11,
- dill>=0.2.6,
- rpy2==2.9.1; python_version>"3.0",
- numpy
- Plotting functionality requires matplotlib>=2.1.0
- with v1.1.0-b1 there are additional dependencies on R related binaries(setup.sh)

pip
~~~~
We recommended that you use a virtual environment to ensure your system is not modified by an installation (virtualenv, conda environment).

::

    pip install -U Skater

conda
~~~~~~
Skater is available in the conda-forge channel with builds for Linux and OS X.
::

    conda install -c conda-forge Skater(the latest version of the library, is not updated on conda)

testing
~~~~~~~~
::

    python -c "from skater.tests.all_tests import run_tests; run_tests()"
