import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from scipy.stats import bootstrap

import time
import cProfile
import pstats
from io import StringIO


"""
This script defines or employs all functions required to perform threshold recalibration.
  - benchmarking compute for the trigger.
  - export logs and metrics.
  - generate time chunks to simulate longitudinal analysis.
  - evaluate the performance metrics for a chunk.
  - load threshold_optimize.py functions.
  
"""

#--------------a benchmarking wrapper to time the trigger-------------#

def benchmark__chunks(func, **kwargs):
    start = time.time()
    result = func(**kwargs)
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    return result


#---------------Export to file recalibration results------------------#


def export_results(results, name='results', metrics_dir='.', log_dir='.', pkl_dir='.'):
    """
    Export metrics and log to CSV, and the full results object to a pickle file.
    File names are appended with the `name` string.

    Params:
    -------
    
    - results: dict with 'metrics' and 'log' keys
    - name: str, identifier to append to filenames
    - metrics_dir, log_dir, pkl_dir: str, directories for each file type

    Returns:
    --------
    metrics: csv
    log: csv
    results: pkl
    
    """
    metrics_path = os.path.join(metrics_dir, f'{name}_chunk_metrics.csv')
    log_path = os.path.join(log_dir, f'{name}_chunk_log.csv')
    pkl_path = os.path.join(pkl_dir, f'{name}_chunks_results.pkl')

    try:
        pd.DataFrame(results['metrics']).to_csv(metrics_path, index=False)
        pd.DataFrame(results['log']).to_csv(log_path, index=False)
        pd.to_pickle(results, pkl_path)
        print(f"Saved metrics to {metrics_path}, log to {log_path}, and results to {pkl_path}")
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to export results for {name}: {e}")


#--------------------Generate data in time chunks----------------------#


def generate_time_chunks(start, end, freq='M'):
    """
    This function generate time-based chunks between start and end dates.

    Parameters
    ----------
    - start: pd.Timestamp, start of the time range
    - end: pd.Timestamp, end of the time range
    - freq: str, frequency string (e.g., 'M', 'W', 'Q'). Default to 'M' months.

    Returns
    -------
    chunks: list (start, end) tuples for each time chunk
    
    """
    # initiallize the chunk list of tuples
    chunks = []
    # start date
    current = start
    # loops until the end date
    while current <= end:
        # create the first chunk
        next_time = (current + pd.tseries.frequencies.to_offset(freq))
        # clamp the end time
        chunk_end = next_time - pd.Timedelta(days=1)
        # condition to identify the end chunk
        if chunk_end > end:
            chunk_end = end
        # append each chunk to chunks tuple
        chunks.append((current, chunk_end))
        # next chunk becomes current chunk in the loop
        current = next_time
    return chunks


#-----------------


def evaluate_metrics(chunk_df, target_prob, target_outcome, thresh, start, end,n_bootstrap=200, random_state=42):
    """
    Calculate evaluation metrics for each chunk and report chunk size and class imbalance ratio.

    Params:
    -------
    
    
    """
    y_true = chunk_df[target_outcome].values
    y_pred = (chunk_df[target_prob] >= thresh).astype(int).values

    def boot_ci(func):
        return bootstrap((y_true, y_pred), statistic=lambda y, yp: func(y, yp),
                          vectorized=False, n_resamples=n_bootstrap, method='percentile',
                          random_state=random_state).confidence_interval

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    FDR = (1 - precision_score(y_true, y_pred, zero_division=0))
    positive = np.sum(y_true)
    negative = len(chunk_df) - positive
    cir = positive / negative if negative > 0 else np.nan

    # Safe fallback in case of too few samples and stable predictions
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    if (
        len(unique_true) > 1 and
        len(unique_pred) > 1 and
        positive > 1 and
        negative > 1
    ):
        ci_precision = tuple(boot_ci(lambda y, yp: precision_score(y, yp, zero_division=0)))
        ci_recall = tuple(boot_ci(lambda y, yp: recall_score(y, yp, zero_division=0)))
        ci_FDR = tuple(boot_ci(lambda y, yp: (1 - precision_score(y_true, y_pred, zero_division=0))))
    else:
        ci_precision = (np.nan, np.nan)
        ci_recall = (np.nan, np.nan)
        ci_FDR = (np.nan, np.nan)

    return {
        'start': start,
        'end': end,
        'threshold': thresh,
        'count': len(chunk_df),
        'CIR': cir,
        'precision': precision,
        'ci_precision': ci_precision,
        'recall': recall,
        'ci_recall': ci_recall,
        'FDR': FDR,
        'ci_FDR': ci_FDR
    }


#---------------------recalibration function----------------------#


def apply_threshold_to_chunks(
    df,
    date_col,
    target_prob,
    target_outcome,
    threshold,
    prior_metrics,
    freq='M',
    n_bootstrap=200,
    tolerance=0.05,
    cutoff_recall=0.8,
    cutoff_precision=0.33,
    lookback_type=None,
    lookback_value=None,
    stratify_by=None,
    hyperparams=None
):
  """
  This function performs threshold recalibration.
  Functions required:
  - generate_time_chunks
  - evaluate_metrics
  - threshold_optimizer
    - load threshold_optimize script.

  Params:
  -------
  df: pd.DataFrame
    Desc: Data to run optimization and calibration on.
  date_col: str
    Desc: Takes as input a str denoting the datetime column.
  target_prob: str
    Desc: Takes as input a str denoting the model's predicted probabilities.
  target_outcome: str
    Desc: Takes as input a str denoting the model's actual outcome.
  threshold: int (float)
    Desc: Takes the most recent optimized decision threshold (probability).
  
    
  """
    # check timestamp type, convert if necessary
    if pd.api.types.is_datetime64tz_dtype(df[date_col]):
        print(f"[INFO] Converting timezone-aware datetime in '{date_col}' to naive timestamps.")
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None)
    # generate_time_chunks to obtain tuples of start,end dates for each chunk
    chunks = generate_time_chunks(df[date_col].min(), df[date_col].max(), freq=freq)
    
    # initialize the chunk_metrics list to store output for each month
    chunk_metrics = []
    current_threshold = threshold
    
#-------------enter the dragon!--------------#
    # loop over the tuples of start/end date chunks
    for start, end in chunks:
        chunk_df = df[(df[date_col] >= start) & (df[date_col] <= end)]
        if stratify_by and stratify_by not in chunk_df.columns and stratify_by in df.columns:
            chunk_df[stratify_by] = df.loc[chunk_df.index, stratify_by]

        original_chunk_df = chunk_df.copy()
        temp_start = start
        extended_window_triggered = False

        if lookback_type and len(chunk_df) < 10:
            extended_window_triggered = True
            while True:
                if lookback_type == 'days':
                    temp_start -= pd.Timedelta(days=lookback_value)
                elif lookback_type == 'weeks':
                    temp_start -= pd.Timedelta(weeks=lookback_value)
                elif lookback_type == 'chunks':
                    temp_start -= (end - start) * lookback_value
                chunk_df = df[(df[date_col] >= temp_start) & (df[date_col] <= end)]
                if len(chunk_df) >= 10 or temp_start < df[date_col].min():
                    break

        metrics = evaluate_metrics(original_chunk_df, target_prob, target_outcome, current_threshold, start, end, n_bootstrap)
        if metrics is None:
            print(f"[WARNING] Evaluation failed for chunk {start} to {end}. Skipping.")
            continue

        metrics['original_chunk_start'] = start
        metrics['original_chunk_end'] = end
        metrics['effective_chunk_start'] = temp_start if extended_window_triggered else start
        metrics['effective_chunk_end'] = end
        metrics['extended_window'] = extended_window_triggered

        recall_baseline = prior_metrics['recall'] if isinstance(prior_metrics, dict) else prior_metrics
        recall_diff = metrics['recall'] - recall_baseline
        metrics['recall_diff'] = recall_diff
        metrics['recall_baseline'] = recall_baseline

        recall_trigger = metrics['recall'] < cutoff_recall
        precision_trigger = metrics['precision'] < cutoff_precision
        triggered_by = []
        if recall_trigger:
            triggered_by.append('recall')
        if precision_trigger:
            triggered_by.append('precision')

        if recall_trigger or precision_trigger:
            percent_change = abs(recall_diff) / recall_baseline if recall_baseline else 0.0
            metrics['recall_percent_change'] = percent_change
            metrics['recalibrate'] = False
            metrics['triggered_by'] = triggered_by

            if percent_change > tolerance or precision_trigger:
                recal_result = threshold_optimizer(
                    use_lookback_if_insufficient=True,
                    lookback_window_days=lookback_value if lookback_type == 'days' else 30,
                    data=chunk_df,
                    target_prob=target_prob,
                    target_outcome=target_outcome,
                    stratify_by=None,
                    thresholds=np.linspace(0.01, 0.5, 50),
                    hyperparams=None
                )
                print(f"[DEBUG] Stratified keys in recalibration: {list(recal_result.get('results', {}).keys())}")

                global_result = recal_result.get('results', {}).get('global')
                if global_result is None:
                    print(f"[WARNING] Recalibration failed due to missing 'global' results.")
                    chunk_metrics.append(metrics)
                    prior_metrics = metrics['recall']
                    continue

                best_thresh = global_result[1].get('fbeta', {}).get('threshold')
                current_threshold = best_thresh

                new_metrics = evaluate_metrics(original_chunk_df, target_prob, target_outcome, best_thresh, start, end)
                new_metrics.update({
                    'recall_baseline': recall_baseline,
                    'recall_diff': new_metrics['recall'] - recall_baseline,
                    'recall_percent_change': abs(new_metrics['recall'] - recall_baseline) / recall_baseline if recall_baseline else 0.0,
                    'original_chunk_start': start,
                    'original_chunk_end': end,
                    'effective_chunk_start': temp_start,
                    'effective_chunk_end': end,
                    'extended_window': extended_window_triggered,
                    'recalibrated_chunk_start': start,
                    'recalibrated_chunk_end': end,
                    'recalibrate': True,
                    'recalibration_t_curve': global_result[0],
                    'recalibration_best_metrics': global_result[1],
                    'recalibrated_threshold': best_thresh,
                    'lookback_used_in_optimizer': True,
                    'lookback_window_days': lookback_value if lookback_type == 'days' else 30,
                    'triggered_by': triggered_by
                })
                metrics = new_metrics
                prior_metrics = metrics['recall']
        else:
            metrics['recall_percent_change'] = 0.0
            metrics['recalibrate'] = False
            metrics['triggered_by'] = []
            prior_metrics = metrics['recall']

        if 'recalibrate' not in metrics:
            metrics['recalibrate'] = False

        chunk_metrics.append(metrics)

    return {
        'metrics': chunk_metrics,
        'log': [
            {
                'event': 'recalibration' if m['recalibrate'] else 'evaluation',
                'start': m['original_chunk_start'],
                'end': m['original_chunk_end'],
                'effective_start': m['effective_chunk_start'],
                'effective_end': m['effective_chunk_end'],
                'threshold_used': m['recalibrated_threshold'] if m['recalibrate'] else threshold,
                'recall': m['recall'],
                'recall_baseline': m['recall_baseline'],
                'recall_diff': m['recall_diff'],
                'recall_percent_change': m['recall_percent_change'],
                'extended_window': m.get('extended_window', False),
                'CIR': m['CIR'],
                'precision': m['precision'],
                'triggered_by': m.get('triggered_by', []),
                'recalibration_t_curve': m.get('recalibration_t_curve'),
                'recalibration_best_metrics': m.get('recalibration_best_metrics'),
            }
            for m in chunk_metrics
        ]
    }
