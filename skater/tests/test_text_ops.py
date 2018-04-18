import unittest
import pandas as pd
import numpy as np

from skater.util.text_ops import cleaner
from skater.util.dataops import convert_dataframe_to_dict


class TestTextOps(unittest.TestCase):

    def setUp(self):
        pass


    def test_cleaner(self):
        raw_text = "    11111this is just an example..., don't be surprised!!!0000000    "
        expected_result = "this is just an example , don t be surprised"
        result = cleaner(raw_text, to_lower=True, norm_num=False, char_to_strip="0|' '|1")
        self.assertEquals(result, expected_result)


    def test_utils(self):
        # converting of dataframe to dict
        data = np.array([['', 'feature_name', 'relevance_scores'],
                         ['R1', 'F1', 20],
                         ['R2', 'F2', 40]])
        test_df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        test_dict = convert_dataframe_to_dict("feature_name", "relevance_scores", test_df)
        self.assertEquals(test_dict['F1'], '20')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextOps)
    unittest.TextTestRunner(verbosity=2).run(suite)
