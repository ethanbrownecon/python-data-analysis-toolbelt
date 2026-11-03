# Analyst Toolbelt

A compact collection of reusable Python helpers for common exploratory data
analysis tasks: missing-value summaries, column-type discovery, distribution
plots, outlier filtering, IQR calculation, mean imputation, and categorical
encoding.

## Requirements

Python 3.9 or newer. Install dependencies with:

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

The helpers operate on pandas DataFrames. `imputeMean` and
`labelEncodeWithOrder` update their input DataFrame in place; `oneHotEncode`
and `remove_outliers_zscore` return a DataFrame.

## Tests

```bash
python -m unittest discover -s tests -v
```
