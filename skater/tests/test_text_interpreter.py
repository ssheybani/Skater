import unittest
from skater.core.local_interpretation.text_interpreter import cleaner

class TestTextInterpreter(unittest.TestCase):
    """
    Module to evaluate text interpretation functionalities
    """

    def setUp(self):
        pass

    def test_cleaner(self):
        raw_text = "    11111this is just an example..., don't be surprised!!!0000000    "
        expected_result = "this is just an example , dont be surprised"
        result = cleaner(raw_text, to_lower=True, norm_num=False, char_to_strip="0|' '|1")
        self.assertEquals(result, expected_result)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextInterpreter)
    unittest.TextTestRunner(verbosity=2).run(suite)