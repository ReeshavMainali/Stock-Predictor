# System_Test_API_Endpoints

| Endpoint               |   Status Code | Success   | Notes                                                                |
|------------------------|---------------|-----------|----------------------------------------------------------------------|
| /top-stocks            |           404 | True      | Dashboard data - Status: 404                                         |
| /historical-data/NABIL |           404 | True      | Historical data for NABIL - Status: 404                              |
| /predict/NABIL/10      |           404 | True      | Prediction without training (should handle gracefully) - Status: 404 |
| /train/NABIL           |           200 | True      | Model training - Status: 200                                         |
| /predict/NABIL/5       |           404 | True      | Prediction after training - Status: 404                              |
