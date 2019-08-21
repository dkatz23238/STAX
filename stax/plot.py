import argparse
import pandas as pd
from stax import TimeSeries
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

plt.style.use("ggplot")
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

    parser.add_argument('results',
                        metavar='results',
                        type=str,
                        help='Results of STAX analysis')

    parser.add_argument('title',
                        metavar='title',
                        type=str,
                        help='Title of the plot')

    args = parser.parse_args()
    df = pd.read_csv(args.table)
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    series = df[args.column]

    results = json.loads(open(args.results).read())
    color = iter(cm.rainbow(np.linspace(0, 1, 7)))

    # series.plot()
    split = results["meta"]["train_test_split_index"]
    plt.title(args.title)
    c = next(color)
    plt.plot(series.index[:split], series.values[:split], c=c, alpha=0.4)
    for model in results["models"]:

        p = results["models"][model]["test_predictions"]
        # print(p)
        plt.plot(series.index[split:], p, label=model, c=c)

        # Confidence Intervals
        if results["models"][model]["test_confidence_intervals"] != None:
            lower = [
                i[0]
                for i in results["models"][model]["test_confidence_intervals"]
            ]
            upper = [
                i[1]
                for i in results["models"][model]["test_confidence_intervals"]
            ]
            plt.fill_between(series.index[split:],
                             lower,
                             upper,
                             alpha=0.5,
                             color=c,
                             label=model + "confidence_intervals")
            c = next(color)

    plt.plot(series.index[split:], series.values[split:], label="original")
    plt.axvline(x=series.index[split],
                color="black",
                alpha=0.3,
                linestyle='--')
    plt.legend()
    plt.show()