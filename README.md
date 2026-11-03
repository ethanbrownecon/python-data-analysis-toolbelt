# Analyst Toolbelt

Reusable Python helpers for common exploratory data analysis and cleaning steps, so the same routines do not have to be rewritten for every project. All helpers work on pandas DataFrames.

## What is included

- `printAmtNA`: prints the number of missing values in each column.
- `findCategoricalandNumericalCols`: returns lists of categorical and numerical columns.
- `makeHist` and `get_normal_dist`: plot a column's distribution, with a fitted normal curve for the second.
- `remove_outliers_zscore`: drops rows beyond a z score threshold and reports the shape before and after.
- `calculate_iqr`: returns the interquartile range of a column.
- `imputeMean`: fills missing values with the column mean, in place.
- `labelEncodeWithOrder`: label encodes a categorical column in a specified order, in place.
- `oneHotEncode`: returns the DataFrame with a categorical column one hot encoded.

## Requirements

Python 3.9 or newer.

```bash
python -m pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
import toolbelt as tb

df = pd.DataFrame({"price": [12, 15, 18, 21]})
print(tb.calculate_iqr(df, "price"))
filtered = tb.remove_outliers_zscore(df, "price", threshold=2)
```

`imputeMean` and `labelEncodeWithOrder` modify their input DataFrame. `oneHotEncode` and `remove_outliers_zscore` return a new one.

## Tests

```bash
python -m unittest discover -s tests -v
```
