# legibility-estimation
Replication material for "Estimating The Legibility of International Borders"

## 1. Data

### 1.1 Annotations

### 1.2 Bing Imagery

*For the rest of the instructions, SCRAPE_DIR will be used to denote the directory in which the scrape is located.*

## 2. Training

### 2.1 Baselines
* To compute accuracy scores for the baseline methods, we first compute thresholds for each method using the training data. To do so, navigate to `legibility-estimation/MODEL/binary/threshold` and run `python3 threshold.py -calc_threshold -clustering -distance --scrape_dir SCRAPE_DIR`.

### 2.2 BorderCut and Transfer Models

## 3. Evaluation

### 3.1 Baselines

#### 3.1.1 Accuracy
* Now, to compute accuracy for the baseline methods, again navigate to `legibility-estimation/MODEL/binary/threshold` and run `python3 threshold.py -test_threshold -clustering -distance --scrape_dir SCRAPE_DIR`. The accuracy scores will be written to `testing.csv`.
#### 3.1.2 ROC AUC
* To compute ROC AUC scores for the baseline methods, navigate to `legibility-estimation/MODEL/binary` and run `python3 compute_roc.py -clustering -distance --scrape_dir SCRAPE_DIR`. After this is finished, the results will be in the `roc_auc.csv` file in the same directory.

#### 3.1.3 Other Metrics
* Navigate to `legibility-estimation/MODEL/binary/metrics` and run `python3 compute_metrics.py --score_file ../threshold/test-scores/`. This will print out the metrics to the command line. *Note: This step must be completed after 3.1.1, as the scores that are used in this step are computed there.*

### 3.2 BorderCut and Transfer Models

#### 3.2.1 Accuracy
* To compute the accuracy for one of the transfer/bordercut models provided, navigate to `legibility-estimation/MODEL/binary` and run `python3 transfer_acc.py --model_type TYPE` where *TYPE* is one of [bordercut, vit, resnet].
* To compute the accuracy for a model that you have trained, include the additional argument `--model_path PATH` where *PATH* is the path to the trained model.
* The accuracy for the model will be output in `accuracy.csv` and the validation and test outputs will be in `outputs/MODEL-NAME-{val,test}.csv`.

#### 3.2.2 ROC AUC
* To compute the roc-auc for one of the transfer/bordercut models provided, navigate to `legibility-estimation/MODEL/binary` and run `python3 compute_roc.py --model_type TYPE` where *TYPE* is one of [bordercut, vit, resnet].
* To compute the roc-auc for a model that you have trained, include the additional argument `--model_path PATH` where *PATH* is the path to the trained model.
* The roc-auc's computed will be output in `roc_auc.csv`.

#### 3.2.3 Other Metrics
* Navigate to `legibility-estimation/MODEL/binary/metrics` and run `python3 compute_metrics.py -transfer --score_file ../outputs/MODEL-NAME-{test,val}.csv`, where MODEL-NAME-{test,val}.csv is the name a file produced by 3.2.1. This will print out the metrics to the command line. *Note: This step must be completed after 3.2.1, as the scores that are used in this step are computed there.*
