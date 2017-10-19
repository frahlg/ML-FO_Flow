import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-0.0010604253695587677
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.0001)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=16, min_samples_split=13, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=8, max_features=0.6000000000000001, min_samples_leaf=9, min_samples_split=13, n_estimators=100, subsample=0.9000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
