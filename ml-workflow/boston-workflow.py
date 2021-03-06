#Sample ML Regression Workflow Example
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
scaler = MinMaxScaler()
scaler.fit_transform(X_test)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(),LinearRegression())
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
from sklearn.metrics import explained_variance_score
score = explained_variance_score(y_test, y_pred)
print(score)
from joblib import dump
dump(pipe, 'regression.joblib')
