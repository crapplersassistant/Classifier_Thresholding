import pandas as pd
import pickle
from itertools import combinations
from functools import reduce
import importlib
import datetime

"""
Various and sundry helper functions for analyzing and summarizing the data.
"""

#--------------------Describe a dataset---------------------#

def describe_dataset(df, date_col=None):
    """
    Describes a dataset by calculating various statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze
    date_col : str, optional
        Name of the date column to compute time range
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics for each column
    """
    import pandas as pd

    # Get basic info about data types
    dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
    
    # Calculate missing values and totals for each column
    missing = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    totals = pd.DataFrame(df.count(), columns=['Total Values'])
    
    # Calculate the number of unique values for each column
    unique = pd.DataFrame(df.nunique(), columns=['Unique Values'])
    
    # Combine the basic statistics
    stats = dtypes.join([missing, totals, unique])
    
    # Calculate percentage of missing values
    stats['Missing %'] = (stats['Missing Values'] / len(df) * 100).round(2)
    
    # For categorical columns, add value distribution
    stats['Value Distribution'] = ''
    for col in df.select_dtypes(include=['object']).columns:
        value_counts = df[col].value_counts(normalize=True).head(5)
        value_dist = ', '.join([f"{val} ({pct:.1%})" for val, pct in value_counts.items()])
        stats.loc[col, 'Value Distribution'] = value_dist
    
    # For numeric columns, add basic statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        stats['Min'] = df[numeric_cols].min()
        stats['Max'] = df[numeric_cols].max()
        stats['Mean'] = df[numeric_cols].mean()
        stats['Std'] = df[numeric_cols].std()
    
    # Compute time difference if date_col is provided and valid
    if date_col is not None and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            time_diff = df[date_col].max() - df[date_col].min()
            stats.loc[date_col, 'Time Diff'] = time_diff
        except Exception as e:
            stats.loc[date_col, 'Time Diff'] = f"Error: {e}"

    # Reorder columns for better readability
    col_order = ['Data Type', 'Total Values', 'Missing Values', 'Missing %', 
                 'Unique Values', 'Value Distribution', 'Min', 'Max', 'Mean', 'Std', 'Time Diff']
    stats = stats.reindex(columns=[col for col in col_order if col in stats.columns])
    
    return stats

#-------------------------check overlap between two dfs-----------------------#


def check_column_overlap(df1, col1, df2, col2):
    """
    Checks if values from one DataFrame's column are present in another DataFrame's column.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame containing the source column
    col1 : str
        Name of the column in df1 to check
    df2 : pd.DataFrame
        Second DataFrame containing the target column
    col2 : str
        Name of the column in df2 to check against
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'overlap_count': Number of overlapping values
        - 'total_values': Total number of unique values in df1[col1]
        - 'overlap_percentage': Percentage of values from df1[col1] that are in df2[col2]
        - 'overlapping_values': Set of values that overlap
        - 'missing_values': Set of values from df1[col1] not found in df2[col2]
    """
    # Get unique values from both columns
    values1 = set(df1[col1].unique())
    values2 = set(df2[col2].unique())
    
    # Find overlapping values
    overlapping = values1.intersection(values2)
    missing = values1 - values2
    
    # Calculate statistics
    total_values = len(values1)
    overlap_count = len(overlapping)
    overlap_percentage = (overlap_count / total_values * 100) if total_values > 0 else 0
    
    return {
        'overlap_count': overlap_count,
        'total_values': total_values,
        'overlap_percentage': overlap_percentage,
        'overlapping_values': overlapping,
        'missing_values': missing
    }


#---------------------Modify and extract importance results----------------------#


def extract_metrics_df(results):
    """
    Flatten and filter chunk-level metrics to a single DataFrame.
    Extracts CIR, count, precision, threshold, start dates, and all keys beginning with 'recall_'.
    Also adds a 'recalibrated' flag and sorts by start_date.
    """
    records = []
    for m in results['metrics']:
        row = {
            'CIR': m.get('CIR'),
            'count': m.get('count'),
            'precision': m.get('precision'),
            'threshold': m.get('threshold') or m.get('recalibrated_threshold'),
            'start_date': m.get('original_chunk_start'),
            'end_date': m.get('end'),
            'effective_start_date': m.get('effective_chunk_start'),
            'recalibrated': m.get('recalibrate', False),
            'Recall percent change' : m.get('recall_percent_change')
        }
        # Include all keys beginning with 'recall_'
        row.update({k: v for k, v in m.items() if k.startswith('recall')})
        records.append(row)

    df = pd.DataFrame(records)
    df = df.rename(columns={
    'precision': 'PPV',
    'CIR': 'IR',
    'recall': 'Sensitivity',
    'recall_diff': 'Recall Delta'
    })
    df['Monthly'] = pd.PeriodIndex(df['end_date'], freq='M').to_timestamp()
    return df.sort_values(by='start_date').reset_index(drop=True)


#-----------------------Metrics wide-to-long-------------------#


def reshape_for_metric_plot(df):
    """
    Takes a DataFrame and returns a reshaped version:
    - end_date → Group (YYYY-MM)
    - Sensitivity, PPV → Metric, Value
    - Sorted by Metric
    """
    df_copy = df.copy()
    df_copy['Group'] = pd.to_datetime(df_copy['end_date']).dt.to_period('M').astype(str)
    
    df_long = df_copy.melt(
        id_vars='Group',
        value_vars=['Sensitivity', 'PPV'],
        var_name='Metric',
        value_name='Value'
    )

    return df_long.sort_values(by='Metric').reset_index(drop=True)
