import unittest
import pandas as pd
from skater.core.global_interpretation.rule_list import SBRL


class TestValidation(unittest.TestCase):
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestValidation)
    unittest.TextTestRunner(verbosity=2).run(suite)