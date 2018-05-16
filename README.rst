.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>

Skater
=======
Skater is a unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable
machine learning system often needed for real world use-cases(** we are actively working towards to enabling faithful interpretability for all forms models). It is an open source python library designed to
demystify the learned structures of a black box model both globally(inference on the basis of a complete data set)
and locally(inference about an individual prediction). 

The project was started as a research idea to find ways to enable better interpretability(preferably human interpretability) to predictive "black boxes" both for researchers and practioners. The project is still in beta phase.

.. image:: https://travis-ci.org/datascienceinc/Skater.svg?branch=master
    :target: https://travis-ci.com/datascienceinc/Skater
    :alt: Build Status

.. image:: https://coveralls.io/repos/github/datascienceinc/Skater/badge.svg?branch=master
    :target: https://coveralls.io/github/datascienceinc/Skater?branch=master
    
    
HighLevel Design 
================
.. image:: https://github.com/datascienceinc/Skater/blob/master/presentations/designs/interpretable_mls.png


📖 Documentation
================

=================== ===
`Overview`_         Introduction to the Skater library
`Installing`_       How to install the Skater library
`Tutorial`_         Steps to use Skater effectively.
`API Reference`_    The detailed reference for Skater's API.
`Contributing`_     Guide to contributing to the Skater project.
`Examples`_         Interactive notebook examples
=================== ===

.. _Overview: https://datascienceinc.github.io/Skater/overview.html
.. _Installing: https://datascienceinc.github.io/Skater/install.html
.. _Tutorial: https://datascienceinc.github.io/Skater/tutorial.html
.. _API Reference: https://datascienceinc.github.io/Skater/api.html
.. _Examples: https://datascienceinc.github.io/Skater/gallery.html
.. _Contributing: https://github.com/datascienceinc/Skater/blob/master/CONTRIBUTING.rst

💬 Feedback/Questions
=====================

=========================  ===
**Feature Requests/Bugs**  `GitHub issue tracker`_
**Usage questions**        `Gitter chat`_
**General discussion**     `Gitter chat`_
=========================  ===

.. _GitHub issue tracker: https://github.com/datascienceinc/Skater/issues
.. _Gitter chat: https://gitter.im/datascienceinc-skater

Install Skater
==============
For detailed information on the dependencies and intallation instruction check out `installing skater
<https://datascienceinc.github.io/Skater/install.html>`_.

pip
~~~
::

    Option 1: without rule lists 
    pip install -U Skater
    
    Option 2: with rule lists
    1. conda install gxx_linux-64
    2. sudo pip install -U --no-deps --force-reinstall --install-option="--rl=True" skater


To get the latest changes try cloning the repo and use the below mentioned commands to get started,
::
    Option 1: without rule lists
    1. git clone the repo
    2. sudo python setup.py install
    
    Option 2: with rule lists
    1. conda install gxx_linux-64
    2. sudo pip install -U --no-deps --force-reinstall --install-option="--rl=True" skater==1.1.0

Testing
~~~~~~~
1. If repo is cloned:
::
    python skater/tests/all_tests.py

2. If pip installed:
::
    python -c "from skater.tests.all_tests import run_tests; run_tests()"


Usage and Examples
==================
Since the project is under active development, the best way to understand usage would be to follow the examples mentioned in the `Gallery of Interactive Notebook <https://datascienceinc.github.io/Skater/gallery.html>`_.
 
Algorithms
~~~~~~~~~~
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Scope of Interpretation |            Algorithms                               |
+=========+=========+=====+===========+===========+==============+==============+
| Global Interpretation   | Model agnostic Feature Importance                   | 
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Global Interpretation   | Model agnostic Partial Dependence Plots             |
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Local Interpretation    | Local Interpretable Model Explanation(LIME)         |
+---------+---------+-----+-----------+-----------------------------------------+
| Local Interpretation    | DNNs      | - Layer-wise Relevance Propagation      |
|                         |           |   (e-LRP): image                        |
|                         |           |                                         |
|                         |           | - Integrated Gradient: image and text   |
|                         |           |                                         |
+---------+---------+-----+-----------+-----------------------------------------+
| Global and Local        | Scalable Bayesian Rule Lists                        |
| Interpretation          |                                                     |
+---------+---------+-----+-----------+-----------+--------------+--------------+
 


Citation
========
If you decide to use Skater to resolve interpretability needs, please consider citing the project with the below mentioned DOI,
::
    @misc{pramit_choudhary_2018_1198885,
      author       = {Pramit Choudhary and
                      Aaron Kramer and
                      datascience.com team, contributors},
      title        = {{Skater: Model Interpretation Library}},
      month        = mar,
      year         = 2018,
      doi          = {10.5281/zenodo.1198885},
      url          = {https://doi.org/10.5281/zenodo.1198885}
    }

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1198885.svg
   :target: https://doi.org/10.5281/zenodo.1198885


R Client
========
Refer to https://github.com/christophM/iml 

   
Books
=====
1. Christoph Molnar, Interpretable Machine Learning: https://christophm.github.io/interpretable-ml-book/ 
2. Dipanjan Sarkar et al., Practical Machine Learning with Python: https://github.com/dipanjanS/practical-machine-learning-with-python
