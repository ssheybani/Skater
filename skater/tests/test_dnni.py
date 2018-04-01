import pandas as pd
import numpy as np
import unittest
import sys

class TestRuleList(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
    unittest.TextTestRunner(verbosity=2).run(suite)

