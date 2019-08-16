# STAX: Automated Time Series Forecasting Tool

```
python -m stax [-h] table column frequency output
```
### Command Line Arguments
- `table`:  refers to the csv file with the time series data
- `column`:  which column to forecast
- `frequency`:  daily or monthly data
- `output`:  full directory for JSON results

# Example Output

```json
{
    "meta": {
        "train": {
            "values": [
                112,
                ...
                407
            ]
        },
        "test": {
            "values": [
                362,
                405,
                ...
                432
            ]
        }
    },
    "models": {
        "ARIMA": {
            "model": "ARIMA",
            "test_predictions": [
                363.2909859213608,
                369.8852769813491,
                ...
                484.92738208204236
            ],
            "test_confidence_intervals": [
                [
                    323.8192989290492,
                    402.7626729136724
                ],
                ...
            ],
            "test_mean_absoloute_percent_error": 0.0949
        },
        "ExponentialSmoothing": {
            "model": "TripleExponentialSmoothing",
            "test_predictions": [
                343.11984637306654,
                ...
            ],
            "test_confidence_intervals": null,
            "test_mean_absoloute_percent_error": 0.0786
        }
    }
}
```