import stax
import pandas as pd

# Test Time Series Instance
def test_time_series_instance_jobs():
    """ Test that you can instantiate a series. """
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)
    assert isinstance(ts, stax.TimeSeries)
    
# Test Bad Time Series Can't Be Created
def test_bad_time_series_instance():
    """ Test that index of series has to be a datetime """
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = range(df.shape[0])
    JOBS = df.Jobs
    try:
        ts = stax.TimeSeries(JOBS, "monthly", 0.8)
    except Exception as e:
        assert isinstance(e, AssertionError)
    
# Test Models Work, Create Predicatble Results
def test_ETS_model_jobs():
    """ Test ETS model training. """
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)
    model, test_pred, test_conf, test_metrics, OOS_pred, OOS_conf = stax.train_expsmoothing(ts)
    assert model.aic < 1420
    assert test_conf == None
    assert test_metrics[0]['mean_absolute_percent_error'] < 0.05
    assert len(OOS_pred) == 12
    assert len(test_pred) == 18

def test_ARIMA_model_jobs():
    """Test ARIMA model training. l"""
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)
    model, test_pred, test_conf, test_metrics, OOS_pred, OOS_conf = stax.train_arima(ts)
    assert model.aic() < 1800
    assert len(test_conf) == 18
    assert test_metrics[0]['mean_absolute_percent_error'] < 0.05
    assert len(OOS_pred) == 12
    assert len(test_pred) == 18
    

def test_TBATS_model_jobs():
    """Test TBATS model training."""
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)
    model, test_pred, test_conf, test_metrics, OOS_pred, OOS_conf = stax.train_tbats(ts)
    assert model.aic < 1900
    assert isinstance(test_conf, list)
    assert test_metrics[0]['mean_absolute_percent_error'] < 0.05
    assert len(OOS_pred) == 12
    assert len(test_pred) == 18

# Test Statistics Work, Create Predictable Results
def test_ACF_jobs():
    """Test ACF and PACF functionality."""
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)

    ACF = stax.tools.ACF(ts)
    PACF = stax.tools.PACF(ts)

    assert len(PACF) == 41
    assert len(ACF) == 41

def test_seasonal_decomp_jobs():
    """Test a seasonal decomposition."""
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)

    decomp = stax.tools.decompose_series(ts)

    assert decomp["method"] == "multiplicative"
    assert isinstance(decomp["trend"], list)
    assert isinstance(decomp["seasonal"], list)
    assert isinstance(decomp["resid"], list)


def test_train_all_models_jobs():
    """Test .train_models() method and .calculate_statistics()."""
    df = pd.read_csv('../data/jobs-trend.csv').set_index("Date")
    df.index = pd.to_datetime(df.index)
    JOBS = df.Jobs
    ts = stax.TimeSeries(JOBS, "monthly", 0.8)

    ts.train_models()
    ts.calculate_statistics()

    assert sorted(list(ts.experiment_results["models"].keys())) == ['ARIMA', 'ExponentialSmoothing', 'TBATS']
    assert sorted(ts.experiment_results["seasonal_decomposition"].keys()) == ['method', 'resid', 'seasonal', 'trend']
    assert sorted(ts.experiment_results["autocorrelation"].keys())  ==  ['ACF', 'PACF']

