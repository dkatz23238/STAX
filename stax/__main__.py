import argparse
import pandas as pd
from stax import TimeSeries
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run time series predictions')
    parser.add_argument('table',
                        metavar='table',
                        type=str,
                        help='CSV file to process')

    parser.add_argument('column',
                        metavar='column',
                        type=str,
                        help='Columns to predict')

    parser.add_argument('frequency',
                        metavar='frequency',
                        type=str,
                        help='Frequency of seasonal trend')

    parser.add_argument('test_split',
                        metavar='test_split',
                        type=float,
                        help='Proportion of train data to test data')

    parser.add_argument('output',
                        metavar='output',
                        type=str,
                        help='Destination for json file')

    args = parser.parse_args()
    df = pd.read_csv(args.table)
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    series = df[args.column]
    ts = TimeSeries(series, args.frequency, train_test_split=args.test_split)

    # Calculate statistics
    ts.calculate_statistics()

    # Train and select models
    ts.train_models()

    data = ts.experiment_results
    data["meta"]["CLI_args"] = {
        "table": args.table,
        "column": args.column,
        "frequency": args.frequency,
        "test_split": args.test_split,
    }
    # Convert Models To String
    for k in data["models"]:
        data["models"][k]["model"] = str(data["models"][k]["model"])

    j = json.dumps(data, indent=4)
    with open(args.output, 'w') as f:
        f.write(j)