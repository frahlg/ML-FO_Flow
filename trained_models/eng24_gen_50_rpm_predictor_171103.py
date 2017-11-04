import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-0.0141909620121556
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=19, n_estimators=100, nthread=1, subsample=0.8500000000000001)),
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=4, min_child_weight=19, n_estimators=100, nthread=1, subsample=0.2)),
            PCA(iterated_power=5, svd_solver="randomized"),
            ElasticNetCV(l1_ratio=0.15000000000000002, tol=0.001)
        ))
    ),
    SelectPercentile(score_func=f_regression, percentile=94),
    ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=4, min_samples_split=11, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
