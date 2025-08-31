# Unit_Test_Preprocessing

| Test                         | Result   | Details                                                    |
|------------------------------|----------|------------------------------------------------------------|
| DataFrame is not empty       | True     | Shape: (200, 4)                                            |
| DataFrame has 'rate' column  | True     | Columns: ['transaction_date', 'rate', 'volume', 'trades']  |
| DataFrame has datetime index | False    | Index type: <class 'pandas.core.indexes.range.RangeIndex'> |
| No missing values in rate    | True     | NaN count: 0                                               |
