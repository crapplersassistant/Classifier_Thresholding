# Classifier Thresholding
Adaptive Prediction Thresholding for Classifiers in Deployment (APTCID)

**We propose a post-training probability threshold recalibration procedure for probabilistic binary or multi-label classifier algorithms in deployment: Adaptive Prediction Thresholding for Classifiers in Deployment (APTCID).  APTCID serves several purposes: it mitigates the burden of updating the model, it maintains goal-driven relevant evaluation metrics (here clinical metrics), and it provides the opportunity to automatically update prediction thresholds and monitors changes in performance.**

![Schematic for Threshold Optimization](Slide54.jpg)

Schematics describing the process of threshold optimization and recalibration.  
- **Step I** Stratification variables, timestamps, prediction probabilities, and outcomes are fed into the pipeline.
- **Step II-III** Our sampling algorithm scans prediction thresholds and computes our metrics.
-   **Step IV** Calculate the maximum median scoring metric, and choose that threshold as the optimal threshold.
-   **Step V** Monitor the performance over time for decay, and if decay is detected, the process repeats from step II.

#### Review Evaluation Metrics (in context)
*Classifiers* output a probability of an outcome...


*Decision threshold* establishes the predicted outcome....
$\mathcal{T_a}, \forall{a} \in \mathbb{R} = [0,1]$

Comparison of predicted outcome to actual outcome is fundamental to model assessment and the basis for evaluation metrics.  We derive these metrics – precision (1), recall (2), accuracy (3), etc. – from the confusion matrix (Figure 1, step III).   

|   	|act  |   	|
|---	|---	|---	|
|pred | TP	| FP	|
|   	| FN	| TN	|

As it was alluded to in the introduction, there has been robust reporting in the literature pertaining to the choice of appropriate model evaluation metrics[8-11].  These works highlight common themes encountered in ML evaluation: inevitable changes in class or outcome distributions, non-constant class imbalance ratios, and the importance of goal-driven metrics. 

Our requirements for model performance are based on the real-world scenario important to clinical outcomes, i.e. goal-driven metrics. Two concerns: primary - minimize false negatives (*FN*), secondary - maintaining an acceptable level of false positives (*FP*). Thus, we chose to track model *precision* (1) and *recall* (2) during deployment.


$Precision(t) = \frac{TP_t}{TP_t + FP_t} \tag{1}$

$Recall(t) = \frac{TP_t}{TP_t + FN_t} \tag{2}$

While one might propose threshold optimization and recalibration that imposes tolerances to achieve professional expectations of the above metrics individually, there is an established scoring metric that combines both into a single score, the *F-measure*. The harmonic mean between sensitivity and precision bounded between zero and one.  This measure can be modified by a positive integer weighting factor, *beta* (3), where a weight greater than one favors sensitivity and a weight less than one favors precision[12].   When an F-measure's beta equals one, it indicates a perfect balance between these metrics(4-6).


![F-beta score formula]()

