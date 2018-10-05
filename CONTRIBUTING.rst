
.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>


Contributing to Skater
===========
Skater is an open source project under the MIT license. We invite
users to help contribute to the project by reporting bugs, requesting features, working
on the documentation and codebase.

.. contents:: Types of Contributions

Reporting a bug
---------------
As with any GitHub project, one should always begin addressing a bug by searching through existing issues.
If an issue for the bug does not exist, please create one with the relevant tags:

=================== ===
Performance         Memory and speed related issues.
Installation        Issues experienced attempting to install Skater.
Plotting            Issues associated with Skater's plotting functionality.
Enhancement         Request for a new feature, or augmentation of an existing feature.
Bug                 Errors or unexpected behavior.
=================== ===

We may augment this tag set as needed.

Submitting a test
-----------------
Currently, the test coverage is low which needs to be improved to guarantee robustness of the solutions supported.
When adding a new funcionality, a subsequent test is desired. 
All test files are under ``skater/tests/``
New tests should be added to relevant test file. If no current file covers
the feature, please add a new file with the following structure:

::

    class MyNewTestClass(unittest.TestCase):
        def setUp(self):
            # allows one to define instructions that need to be executed before each test method.
        
        @classmethod
        def setUpClass(cls):
            # allows one to define instructions that need to be executed before all the tests in an individual class.

        def test_function_1(self):
            ...
        def test_function_2(self):
            ...
            
        def tearDown(self):
            # allows one to define instructions that need to be executed after each test method. 
            
        @classmethod    
        def tearDownClass()
            # allows one to define instructions that need to be executed after all tests in an individual class.
        
    if __name__ == '__main__':
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(unittest.makeSuite(TestData))


Contributing Code
-----------------
Skater is distributed under an MIT license. By submitting a pull request for this project,
you agree to license your contribution under the MIT license to this project as well.


Style
~~~~~~~~~~~~~~~~~~~~
Stylistically, contributions should follow PEP8, with the exception that methods
are separated by 2 lines instead of 1.

Pull Requests
~~~~~~~~~~~~~~~~~~~~
Before a PR is accepted, travis builds must pass on all environments, and flake8
tests must all pass. We also use codacy(https://www.codacy.com/) for code reviews(very helpful in identifying anti-patterns)


Dependencies
~~~~~~~~~~~~~~~~~~~~
Every additional package dependency adds a potential installation complexity,
so only dependencies that are critical to the package should be added to the
code base. PRs that involve the addition of a new dependency will be evaluated
by the following criteria:

- Is the application of the dependency isolated, such that removing it is trivial, or
  will it be deeply integrated into the package.
- Does the dependency have known installation issues on common platforms?
- Does the application using the dependency need to be in the Skater package?



.. |Build Status-master| image:: https://api.travis-ci.com/repositories/datascienceinc/Skater.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master
.. |Skater Logo White| image:: https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg


Roadmap
---------------
We'd like to improve the package in a few key ways. The list below
represents aspects we definitely want to address.

======================= ===
Performance             We would like to improve performance where ever possible. Model agnostic algorithms can
                        only be implemented under a "perturb and observe" framework, whereby inputs are selectively
                        chosen, outputs are observed, and metrics, inferences, visualizations are created. Therefore,
                        the bottleneck is always the speed of the prediction function, which we will not have control over.
                        The best way to improve Skater performance is with parallelization (what is the quickest way
                        to execute N function calls) and intelligent sampling (how few function calls can we make/
                        how few observations can we pass to each function).
Algorithms              There are other interpretation algorithms we'd like to support. One family of algorithms would
                        fall under the category of "model surrogates", where models are approximated, either locally
                        or globally. These algorithms must be accurate/faithful to the original model,
                        and simple/interpretable to be useful. The user would also need to know if and where the surrogate is
                        a poor representative of the original model. Check out the issues sections under tag "algorithm" for 
                        latest set of new algorithms being considered. Create a new issue, if you have another idea.
Plotting                We'd like to iterate on our visualizations to make them more intuitive, and ideally not rely
                        on matplotlib. We have slowly started improving the visualization but lot more work needs to be done.
======================= ===

**We are also in the process of re-defining the roadmap.
