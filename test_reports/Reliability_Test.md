# Reliability_Test

| Test Case                  | Outcome   | Notes                                                                                                                          |
|----------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| Missing data (NaN values)  | Failed    | Exception handling missing data: Found array with 0 sample(s) (shape=(0, 5)) while a minimum of 1 is required by MinMaxScaler. |
| Empty DataFrame            | Passed    | Exception properly raised for empty data: ValueError                                                                           |
| Insufficient training data | Passed    | Exception properly raised for insufficient data: ValueError                                                                    |
| API with invalid symbol    | Passed    | API correctly returned error status: 404                                                                                       |
