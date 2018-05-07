.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>
    <br>
    <br>



**********
Overview
**********

'''''''''''''''''''''''''''''

What is Model Interpretation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The concept of model interpretability in the field of machine learning is still new, largely subjective, and, at times,
controversial. Model interpretation is the ability to explain and validate the decisions of a predictive model to
enable fairness, accountability, and transparency in the algorithmic decision-making
(for a more detailed explanation on the definition of transparency in machine learning, see “Challenges of Transparency” by Adrian Weller).
Or, to state it formally, model interpretation can be defined as the ability to better understand the decision policies
of a machine-learned response function to explain the relationship between independent (input) and dependent (target) variables,
preferably in a human interpretable way. Ideally, you should be able to query the model to understand the "what, why, and how" of
its algorithmic decisions.

Skater
~~~~~~
Skater is a unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable
machine learning system often needed for real world use-cases. It is an open source python library designed to
demystify the learned structures of a black box model both globally(inference on the basis of a complete data set)
and locally(inference about an individual prediction). The library has embraced object-oriented and functional
programming paradigms as deemed necessary to provide scalability and concurrency while keeping code brevity in mind.
The project is still in beta phase and is under active development.

Algorithms supported by Skater:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------+---------+-----+-----------+-----------+--------------+--------------+
| Scope of Interpretation |            Algorithms                               |
+=========+=========+=====+===========+===========+==============+==============+
| Global Interpretation   | Model agnostic Feature Importance                   | 
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Global Interpretation   | Model agnostic Partial Dependence Plots             |
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Local Interpretation    | Local Interpretable Model Explanation(LIME)         |
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Global and Local        | Scalable Bayesian Rule Lists                        |
| Interpretation          |                                                     |
+---------+---------+-----+-----------+-----------+--------------+--------------+

