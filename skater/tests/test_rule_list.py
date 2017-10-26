import unittest

class TestRuleList(unittest.TestCase):

    def setup(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
    unittest.TextTestRunner(verbosity=2).run(suite)