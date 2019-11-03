# STAX: Automated Time Series Forecasting Tool

<p align="center">
  <img width="800" alt="portfolio_view" src="images/logo.png">
</p>

# Command Line Interface

```
python -m stax [-h] table column frequency output
```

| Argument    | Definition                             |
| ----------- | -------------------------------------- |
| `table`     | The csv file with the time series data |
| `column`    | Which column to forecast               |
| `frequency` | Use daily or monthly data              |
| `output`    | Full directory for JSON results        |

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

# Webhook Server with Redis

1. Run the redis server
2. Spin up redis queue workers
3. Start the webhook server
