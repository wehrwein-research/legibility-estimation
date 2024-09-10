# legibility-estimation
Replication material for "Estimating The Legibility of International Borders"

## 1. Data

### 1.1 Annotations
* All of our annotation data has been provided in `legibility-estimation/annotations/`

### 1.2 Bing Imagery
* In order to run the Bing Imagery scrape, we direct you to `https://github.com/wehrwein-research/border-legibility`, where that process is detailed.

*For the rest of the instructions, SCRAPE_DIR will be used to denote the directory in which the scrape is located.*

## 2. Training

### 2.1 Baselines
* To compute accuracy scores for the baseline methods, we first compute thresholds for each method using the training data. To do so, navigate to `legibility-estimation/MODEL/binary/threshold` and run `python3 threshold.py -calc_threshold -clustering -distance --scrape_dir SCRAPE_DIR`.

### 2.2 BorderCut and Transfer Models
* We have provided the model checkpoints for the models used in the paper in the `legibility-estimation/MODEL/bordercut/weights/` directory.
* If you wish to finetune a pretrained ResNet or ViT model, navigate to `legibility-estimation/MODEL/bordercut/` and modify the `vit-config.json` and `resnet-config.json` as necessary. It will be required to put in *SCRAPE_DIR* so that the dataloader can find the imgaes. To train the model and log it in Wandb, run `python3 train_transfer.py {resnet,vit}-config.json yes`, or omit the *yes* to not use Wandb.
* To finetune a pretrained BorderCut model it is a similar process. We have provided a pretrained backbone in the weights directory. However, if you wish to train the BorderCut model from scratch, that process has been detailed in `https://github.com/wehrwein-research/border-legibility`. Next, modify the `bordercut-config.json` as required, specifically, provide paths to the *SCRAPE_DIR* and the pretrained BorderCut model. Then, to train using Wandb, run `python3 bordercut.py bordercut-config.json yes`, or omit the *yes* to not use Wandb.

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
