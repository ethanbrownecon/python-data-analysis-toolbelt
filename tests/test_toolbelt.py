import unittest

import pandas as pd

import toolbelt


class ToolbeltTests(unittest.TestCase):
    def test_column_type_discovery(self):
        frame = pd.DataFrame({"count": [1, 2], "group": ["a", "b"]})
        categorical, numerical = toolbelt.findCategoricalandNumericalCols(frame)
        self.assertEqual(categorical, ["group"])
        self.assertEqual(numerical, ["count"])

    def test_outlier_removal_preserves_index_and_missing_values(self):
        frame = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0, 100.0, None]},
            index=[10, 20, 30, 40, 50],
        )
        result = toolbelt.remove_outliers_zscore(frame, "value", threshold=1)
        self.assertEqual(result.index.tolist(), [10, 20, 30, 50])
        self.assertTrue(pd.isna(result.loc[50, "value"]))

    def test_outlier_removal_keeps_constant_data(self):
        frame = pd.DataFrame({"value": [4.0, 4.0, 4.0]})
        result = toolbelt.remove_outliers_zscore(frame, "value", threshold=1)
        pd.testing.assert_frame_equal(result, frame)

    def test_outlier_threshold_must_be_non_negative(self):
        frame = pd.DataFrame({"value": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            toolbelt.remove_outliers_zscore(frame, "value", threshold=-1)

    def test_iqr(self):
        frame = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        self.assertEqual(toolbelt.calculate_iqr(frame, "value"), 2)

    def test_mean_imputation_updates_dataframe(self):
        frame = pd.DataFrame({"value": [1.0, None, 3.0]})
        toolbelt.imputeMean(frame, "value")
        self.assertEqual(frame["value"].tolist(), [1.0, 2.0, 3.0])

    def test_mean_imputation_rejects_all_missing_column(self):
        frame = pd.DataFrame({"value": [None, None]})
        with self.assertRaises(ValueError):
            toolbelt.imputeMean(frame, "value")

    def test_one_hot_encoding_returns_encoded_copy(self):
        frame = pd.DataFrame({"country": ["CA", "US"]})
        encoded = toolbelt.oneHotEncode(frame, "country")
        self.assertEqual(frame.columns.tolist(), ["country"])
        self.assertEqual(encoded.columns.tolist(), ["country_CA", "country_US"])
        self.assertEqual(encoded.to_numpy().tolist(), [[1, 0], [0, 1]])

    def test_ordered_encoding(self):
        frame = pd.DataFrame({"size": ["small", "large", "medium"]})
        toolbelt.labelEncodeWithOrder(frame, ["small", "medium", "large"], "size")
        self.assertEqual(frame["size"].tolist(), [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
