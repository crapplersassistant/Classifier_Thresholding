"""
This script defines all necessary functions for the optimization procedure:
    
    ***see requirements file for python necessary packages***
    ***handles a user-defined array of thresholds to scan***
    ***handles a user-defined list of dictionaries for hyperparams, if needed***
    
    - precomputed_subsets function for generating n tuples of sampling inthresholds, hyperparamsdices based on fraction of train/test split.
    - subsampling_EM_optimization_with_precomputed_subsets function for generating metric performance at each threshold
    - evaluate_params function only used to optimize the hyperparams
    - validate_inputs function for checking inputs behave as expected
    - generate_subsets function for creating subsets on the data, stratified or not
    - evaluate_stratum function is a wrapper for subsampling_EM_optimization_with_precomputed_subsets that incorporates hyperparams if not None
    - threshold_optimizer function uses the above functions (except evaluate_params) to generate optimization results

"""


#-----------------Precompute Subsets and store indices as tuples----------------------#

def precompute_subsets(y_prob, y_true, n_subsets=200, subset_fraction=0.5, random_state=500):
    """
    Generate tuples of stratified subset indices from predicted probabilities and actual outcomes.
    
    Parameters
    ----------
    y_prob : array-like (float)
        array of model prediction probabilities (mpp) for an outcome.
    y_true : array-like (bool)
        array of actual outcomes associated with mpp.
    n_subsets : int
        User-defined number of subsets to create. The default is 200.
    subset_fraction : float
        Fraction of data split into train and test. The default is 0.5.
    random_state : int
        Set the random number generator seed. The default is 500.

    Raises
    ------
    ValueError
        A quality check to ensure the outcome is binary.

    Returns
    -------
    list
        A list of two element tuples containing the train indices and test indices for subsets.

    """
    # verify sufficient class counts for computing subsets, skip if insufficient
    class_counts = np.bincount(y_true.astype(int), minlength=2)
    if np.any(class_counts < 2):
        warnings.warn(
            "Insufficient class counts for stratified splitting: "
            f"{dict(enumerate(class_counts))}. "
            "Skipping this stratum.",
            RuntimeWarning,
        )
        return None
    # perform stratified sampling with sklearn.model_selection StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(
        n_splits=n_subsets,
        test_size=(1 - subset_fraction),
        random_state=random_state
    )
    return [
        (train_idx, test_idx)
        for train_idx, test_idx in splitter.split(y_prob, y_true)
    ]

#-------------------------Sample, Scan and Score Func---------------------------#

def subsampling_EM_optimization_with_precomputed_subsets(y_prob, y_true, beta=1.0, thresholds=None, subsets=None):
    """
    Sample subsets, scan thresholds, and calculate performance metrics.

    Parameters
    ----------
    y_prob : array-like (float)
        array of model prediction probabilities (mpp) for an outcome.
    y_true : array-like (bool)
        array of actual outcomes associated with mpp.
    beta : int or float
        User-defined beta for weighting the F-measure used for optimization. The default is 1.0 - balanced precision and recall
    thresholds : array-like (float)
        User-defined thresholds to scan using np.linspace - produce n evenly spaced values over some interval. Converted to an np.array by default. The default is None.
    subsets : list (tuples)
        Tuples of train/test subset indices to perform sampling. The default is None.

    Returns
    -------
    dict
        Key-value pairs of thresholds:threshold and corresponding metric:score.

    """
    # convert thresholds to an array (if needed)
    thresholds = np.array(thresholds)
    # generate empty score_map for each metric
    score_maps = {metric: {t: [] for t in thresholds} for metric in [
        'fbeta', 'precision', 'recall', 'kappa', 'jaccard', 'accuracy', 'hamming', 'FNR', 'FDR']}
    # loop over subset indices
    for train_idx, _ in subsets:
        # assign probs and true outcomes from each index
        y_prob_sub = y_prob.iloc[train_idx]
        y_true_sub = y_true.iloc[train_idx]
        # assign predicted outcome based on threshold in thresholds
        y_pred_matrix = (y_prob_sub.to_numpy()[:, None] >= thresholds).astype(int)
        # calculate each metric for each threshold in thresholds and populate score_maps
        for i, t in enumerate(thresholds):
            y_pred = y_pred_matrix[:, i]
            score_maps['fbeta'][t].append(fbeta_score(y_true_sub, y_pred, beta=beta, zero_division=0))
            score_maps['precision'][t].append(precision_score(y_true_sub, y_pred, zero_division=0))
            score_maps['recall'][t].append(recall_score(y_true_sub, y_pred, zero_division=0))
            score_maps['kappa'][t].append(cohen_kappa_score(y_true_sub, y_pred))
            score_maps['jaccard'][t].append(jaccard_score(y_true_sub, y_pred, zero_division=0))
            score_maps['accuracy'][t].append(accuracy_score(y_true_sub, y_pred))
            score_maps['hamming'][t].append(sk_hamming_loss(y_true_sub, y_pred))
            score_maps['FNR'][t].append((1 - recall_score(y_true_sub, y_pred, pos_label=0, zero_division=0)))
            score_maps['FDR'][t].append((1 - precision_score(y_true_sub, y_pred, zero_division=0)))
    # initialize an empty dictionary to store the median metric scores        
    results = {}
    # loop over the metrics in score map
    for metric in score_maps:
        # calculate the median of each metric for each threshold
        results[f'med_{metric}'] = {t: np.median(score_maps[metric][t]) for t in thresholds}
        # calculate the 95%CI intervals of each metric for each threshold
        results[f'ci_{metric}'] = {
            t: (np.percentile(score_maps[metric][t], 2.5), np.percentile(score_maps[metric][t], 97.5))
            for t in thresholds
        }
    # returns a dictionary of median/95%CI at each threshold.
    return {
        'threshold': thresholds,
        'med_fbeta': results['med_fbeta'],
        'med_fbeta 95%CI': results['ci_fbeta'],
        'med_precision (95%CI)': (results['med_precision'], results['ci_precision']),
        'med_recall (95%CI)': (results['med_recall'], results['ci_recall']),
        'med_kappa (95%CI)': (results['med_kappa'], results['ci_kappa']),
        'med_jaccard (95%CI)': (results['med_jaccard'], results['ci_jaccard']),
        'med_accuracy (95%CI)': (results['med_accuracy'], results['ci_accuracy']),
        'med_hamming (95%CI)': (results['med_hamming'], results['ci_hamming']),
        'med_FNR (95%CI)': (results['med_FNR'], results['ci_FNR']),
        'med_FDR (95%CI)': (results['med_FDR'], results['ci_FDR']),
    }

#----------------------------Optimize Hyperparameter(s)---------------------------#

def evaluate_params(params, y_prob, y_true, thresholds, subsets):
    """

    Parameters
    ----------
    params : list (dict)
        A list of dictionaries for each param.
    y_prob : array-like (float)
        array of model prediction probabilities (mpp) for an outcome.
    y_true : array-like (bool)
        array of actual outcomes associated with mpp.
    thresholds : array-like (float)
        User-defined thresholds to scan using np.linspace - produce n evenly spaced values over some interval. Converted to an np.array by default. The default is None.
    subsets : list (tuples)
        Tuples of train/test subset indices to perform sampling. The default is None.

    Returns
    -------
    dict
        key-value pairs of betas, threshold where max-median f-beta, 95%CI of max-median f-beta, and other metrics.
        
    """
    # run sampling function and scan hyperparam(s)
    metrics_result = subsampling_EM_optimization_with_precomputed_subsets(
        y_prob=y_prob,
        y_true=y_true,
        # hyperparam entry
        beta=params['beta'],
        thresholds=thresholds,
        subsets=subsets
    )
    # extract the median f-betas
    median_fbeta = metrics_result["med_fbeta"]
    # and 95%CI
    ci_fbeta = metrics_result["med_fbeta 95%CI"]
    # get the max median f-beta threshold and save
    best_threshold = max(median_fbeta, key=median_fbeta.get)
    # apply best_threshold to y_prob to create y_pred outcome
    y_pred = (y_prob >= best_threshold).astype(int)
    # populate a metrics dictionary at the f-beta optimal threshold
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'jaccard': jaccard_score(y_true, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'FNR': (1 - recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'FDR': (1 - precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        
    }
    # return a dictionary of best performance threshold at each beta for each metric.
    return {
        'beta': params['beta'],
        'best_threshold': best_threshold,
        'max_fbeta': median_fbeta[best_threshold],
        'fbeta_95%CI': ci_fbeta[best_threshold],
        **metrics
    }

#-------------------------------Validate Inputs-------------------------------#

def validate_inputs(data, target_prob, target_outcome, stratify_by, thresholds, hyperparams):
    """
    Raise value errors for improper inputs to threshold_optimizer function.
    
    Params
    ------
    Inputs to threshold_optimizer function.
    
    Returns
    -------
        ValueErrors, if present, for each input
        thresholds: array-like object of threshold values if not previously defined. Default is np.linspace(0.0, 1.0, 101)
    
    """
    if len(data) != len(data[target_outcome]) or len(data) != len(data[target_prob]):
        raise ValueError("data, target_prob, and target_outcome length mismatch. Check data integrity.")
    if target_prob not in data.columns or target_outcome not in data.columns:
        raise ValueError(f"Columns '{target_prob}' or '{target_outcome}' not found in data.")
    if stratify_by and stratify_by not in data.columns:
        raise ValueError(f"stratify_by column '{stratify_by}' not found in data.")
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    elif not isinstance(thresholds, (list, np.ndarray)):
        raise ValueError("thresholds must be a list or numpy array.")
    elif not all(0 <= t <= 1 for t in thresholds):
        raise ValueError("All thresholds must be in the range [0, 1].")
    assert isinstance(param_combinations, list) and isinstance(param_combinations[0], dict)
    return np.array(thresholds)


#---------------------Wrapper for generating stratified subsets----------------------#


def generate_subsets(data, stratify_by, target_prob, target_outcome,
                     use_lookback_if_insufficient=False, lookback_window_days=30,
                     date_col=None):    
    """
    
    A function to generate the tuples of subset indices.  Enables subset generation for stratification variables. Defaults to "global" - all cases- if stratify_by = None.
    Contains testing to ensure subsets are generated properly, and prints how subsets are generated.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame used to optimize or calibrate the decision threshold.
    stratify_by : str
        String denoting the variable by which to stratify.
    target_prob : float
        Data variable of model prediction probabilities (mpp) for an outcome.
    target_outcome : bool
        Data variable of actual outcome associated with mpp.
    use_lookback_if_insufficient : bool
        A conditional argument for dataset augmentation when evaluated time-interval contains < 10 cases. The default is False.
    lookback_window_days : int
        User-defined number of days to lookback to augment threshold recalibration_trigger call to threshold_optimizer. The default is 30.

    Returns
    -------
    dict
        Returns a dictionary of stratified subsets for each keys=unique stratum or key=global if stratify_by = None.

    """
    # test block to check for stratification and generates stratification subsets, otherwise returns global subsets.
    try:
        if stratify_by:
            return {
                val: precompute_subsets(
                    data.loc[data[stratify_by] == val, target_prob],
                    data.loc[data[stratify_by] == val, target_outcome]
                ) for val in data[stratify_by].unique()
            }
        return {"global": precompute_subsets(data[target_prob], data[target_outcome])}
    # if there are insufficient samples, either stratified or global....
    except ValueError as e:
        # set default to False for optimization (we use entire dataset), True in threshold_optimizer call within recalibration_trigger.py
        if use_lookback_if_insufficient:
            # check for date column...
            if date_col is None or date_col not in data.columns:
                raise ValueError("date_col must be provided and exist in the dataframe when using lookback.")
            # alert user to lookback being triggered
            print(f"[INFO] Insufficient class counts, attempting lookback window of {lookback_window_days} days...")
            # assign the date_index (all dates in range) to variable 
            date_index = pd.to_datetime(data[date_col])
            # pull most recent date in date_index using .max()
            latest_date = date_index.max()
            # assign a new start date based on user-defined lookback window (days)
            start_date = latest_date - pd.Timedelta(days=lookback_window_days)
            # index new start date and original end date
            mask = (date_index >= start_date) & (date_index <= latest_date)
            # create the expanded dataset with new start date.
            expanded_data = data.loc[mask]
            # test the expanded data for sufficient sample size.
            try:
                return {"global": precompute_subsets(expanded_data[target_prob], expanded_data[target_outcome])}
            # alert user that lookback was unable to create a large enough sample size
            except ValueError:
                print("[WARNING] Still insufficient class balance after applying lookback window.")
                return {"global": None}
        else:
            raise


#-----------------------wrapper for sampling with stratification--------------------#


def evaluate_stratum(stratum_data, target_prob, target_outcome, thresholds, hyperparams, subset):
    
    """
    Wrapper function for the sampling procedure.
    Used within threshold_optimizer function.
    Contains conditionals for stratification with hyperparameter tuning.
    

    Parameters
    ----------
    stratum_data : dict
        Dictionary of subset indices (values) for each stratum (key).
    target_prob : array-like (float)
        array of model prediction probabilities (mpp) for an outcome.
    target_outcome : array-like (bool)
        array of actual outcomes associated with mpp.
    thresholds : array-like (float)
        User-defined thresholds to scan using np.linspace - produce n evenly spaced values over some interval. Converted to an np.array by default. The default is None.
    hyperparams : list (dict)
        A list of dictionaries for each param.
    subset : list (tuples)
        Tuples of train/test subset indices to perform sampling with or without stratification. The default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame of thresholds and median metrics/CI.
    dict
        Dictionary of best threshold for each metric of interest.

    """
    # check to see whether we need to tune strata for best hyperparam
    if hyperparams is None:
        # return the metrics dict from subsampling function
        metrics = subsampling_EM_optimization_with_precomputed_subsets(
            y_prob=stratum_data[target_prob],
            y_true=stratum_data[target_outcome],
            thresholds=thresholds,
            subsets=subset
        )
        # convert the metrics dict to a pandas dataframe for downstream analysis
        df = pd.DataFrame({
            'threshold': thresholds,
            'median_fbeta': [metrics['med_fbeta'][t] for t in thresholds],
            'ci_fbeta': [metrics['med_fbeta 95%CI'][t] for t in thresholds],
            'median_precision': [metrics['med_precision (95%CI)'][0][t] for t in thresholds],
            'ci_precision': [metrics['med_precision (95%CI)'][1][t] for t in thresholds],
            'median_recall': [metrics['med_recall (95%CI)'][0][t] for t in thresholds],
            'ci_recall': [metrics['med_recall (95%CI)'][1][t] for t in thresholds],
            'median_kappa': [metrics['med_kappa (95%CI)'][0][t] for t in thresholds],
            'ci_kappa': [metrics['med_kappa (95%CI)'][1][t] for t in thresholds],
            'median_jaccard': [metrics['med_jaccard (95%CI)'][0][t] for t in thresholds],
            'ci_jaccard': [metrics['med_jaccard (95%CI)'][1][t] for t in thresholds],
            'median_accuracy': [metrics['med_accuracy (95%CI)'][0][t] for t in thresholds],
            'ci_accuracy': [metrics['med_accuracy (95%CI)'][1][t] for t in thresholds],
            'median_hamming': [metrics['med_hamming (95%CI)'][0][t] for t in thresholds],
            'ci_hamming': [metrics['med_hamming (95%CI)'][1][t] for t in thresholds],
            'median_FNR': [metrics['med_FNR (95%CI)'][0][t] for t in thresholds],
            'ci_FNR': [metrics['med_FNR (95%CI)'][1][t] for t in thresholds],
            'median_FDR': [metrics['med_FDR (95%CI)'][0][t] for t in thresholds],
            'ci_FDR': [metrics['med_FDR (95%CI)'][1][t] for t in thresholds]
        })
        # calculate the best performance threshold for each metric for downstream analysis
        best_metrics = {
            'fbeta': df.loc[df['median_fbeta'].idxmax(), ['median_fbeta', 'ci_fbeta', 'threshold']].to_dict(),
            'precision': df.loc[df['median_precision'].idxmax(), ['median_precision', 'ci_precision', 'threshold']].to_dict(),
            'recall': df.loc[df['median_recall'].idxmax(), ['median_recall', 'ci_recall', 'threshold']].to_dict(),
            'kappa': df.loc[df['median_kappa'].idxmax(), ['median_kappa', 'ci_kappa', 'threshold']].to_dict(),
            'jaccard': df.loc[df['median_jaccard'].idxmax(), ['median_jaccard', 'ci_jaccard', 'threshold']].to_dict(),
            'accuracy': df.loc[df['median_accuracy'].idxmax(), ['median_accuracy', 'ci_accuracy', 'threshold']].to_dict(),
            'hamming': df.loc[df['median_hamming'].idxmin(), ['median_hamming', 'ci_hamming', 'threshold']].to_dict(),
            'FNR': df.loc[df['median_FNR'].idxmin(), ['median_FNR', 'ci_FNR', 'threshold']].to_dict(),
            'FDR': df.loc[df['median_FDR'].idxmin(), ['median_FDR', 'ci_FDR', 'threshold']].to_dict(),

        }

        return df, best_metrics
    # if hyperparams not None, tune on stratified subsets and return best beta.
    return pd.DataFrame([
        evaluate_params(
            params,
            stratum_data[target_prob],
            stratum_data[target_outcome],
            thresholds,
            subset
        ) for params in hyperparams
    ])


#-------------------------Threshold Optimizer Function--------------------------#


def threshold_optimizer(data, target_prob, target_outcome, stratify_by=None, thresholds=None, hyperparams=None, use_lookback_if_insufficient=False, lookback_window_days=30):
    """
    

    Parameters
    ----------
    data : pd.DataFrame
        Dataset that needs to be optimized.  Either optimization set or time-intervals.
    target_prob : array-like (float)
        array of model prediction probabilities (mpp) for an outcome.
    target_outcome : array-like (bool)
        array of actual outcomes associated with mpp.
    stratify_by : str
        Column in data upon which to stratify optimization. The default is None.
    thresholds : array-like (float)
        User-defined thresholds to scan using np.linspace - produce n evenly spaced values over some interval. Converted to an np.array by default. The default is None.
    hyperparams : list (dict)
        A list of dictionaries for each param.
    use_lookback_if_insufficient : bool
        A boolean value indicated whether we want to use lookback augmentation. The default is False.
    lookback_window_days : int
        User-defined window (days) to lookback in time for recalibration. The default is 30.

    Returns
    -------
    dict
        Returns dict object that contains a nested dictionary of results for each stratum and specific target outcome.

    """
    # run the data validation function to check data integrity and generate a range of thresholds if not defined.
    thresholds = validate_inputs(data, target_prob, target_outcome, stratify_by, thresholds, hyperparams)
    # run the generate_subsets function to create the tuple of sampling indices stratified or otherwise (global)
    subsets = generate_subsets(data, stratify_by, target_prob, target_outcome, use_lookback_if_insufficient=use_lookback_if_insufficient, lookback_window_days=lookback_window_days)
    # initialize an empty dict object to populate with results
    results = {}
    # loop over stratum if stratify_by not None
    for stratum, subset in subsets.items():
        if subset is None:
            # alert user to insufficient data in subset
            print(f"[WARNING] Skipping '{stratum}' due to insufficient data for stratified sampling.")
            continue
        # conditional statement to generate the data from subsets if statify_by is or is not None
        stratum_data = data if stratify_by is None else data[data[stratify_by] == stratum]
        # run evaluate_stratum to perform stratified sampling and optimization
        # nest results by stratum, if stratify_by is None, results[stratum] = 'global'
        results[stratum] = evaluate_stratum(stratum_data, target_prob, target_outcome, thresholds, hyperparams, subset)
        
    return {
        # outcome we are optimizing
        'target': target_outcome,
        # column used for stratification
        'stratify_by': stratify_by,
        # a pd.DataFrame of thresholds and median/95%CI scores and a dict of best threshold/metric
        'results': results
    }

