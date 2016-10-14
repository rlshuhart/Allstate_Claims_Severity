import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.operators.preprocessors import ZeroCount

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    ZeroCount(),
    make_union(VotingClassifier([("est", ElasticNet(alpha=0.96, l1_ratio=0.81))]), FunctionTransformer(lambda X: X)),
    MinMaxScaler(),
    LinearSVR(C=25.0, dual=True)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
