# Model Card

## Model Details
Prediction task is to determine whether a person makes over 50K a year.
We use a GradientBoostingClassifier using the optimized hyperparameters in scikit-learn 0.23.2
Hyperparameters tuning was realized using GridSearchCV. Optimal parameters used are:

- learning_rate : 0.05
- max_depth : 5
- min_samples_split : 5
- n_estimators : 500

## Intended Use

This model can be used to predict and get the idea of a person's income range which can used to identify if a person qaulifies for certain facilities or services like loan approval etc.

## Training Data

The training data is the census data available at the UCI library. It is the adult.income data from the data folder.
<br />
Link: [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income )

## Evaluation Data

Evaluation data is splitted from the training set itself. The split was done in 80:20 ratio.

## Metrics
Metrices to evaluate the model performance were:

- Precision : 0.7900960827790096
- Recall : 0.6804583068109484
- F-beta score : 0.7311901504787961

## Ethical Considerations

The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.

## Caveats and Recommendations

Extraction was done from the 1994 Census database. The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population.
It is recommended to use the dataset for training purpose on ML classification or related problems.