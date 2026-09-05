"""Train a churn model and save it to disk.

This script is intentionally robust: MLflow integration is optional and
the script will still train and save a local model when MLflow is not
available or not configured.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Optional progress bar
try:
	from tqdm import tqdm
	TQDM_AVAILABLE = True
except Exception:
	TQDM_AVAILABLE = False


# Paths (can be overridden with env vars)
DATA_PATH = os.getenv('DATA_PATH', 'data/telco_customers.csv')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/churn_model.pkl')

# MLflow configuration via environment variables (optional)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '').strip()
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'telco_churn_experiment')
MLFLOW_REGISTER = os.getenv('MLFLOW_REGISTER_MODEL', 'false').lower() == 'true'
MLFLOW_REGISTERED_NAME = os.getenv('MLFLOW_REGISTERED_NAME', 'ChurnModel')

# Import mlflow only if available - don't fail when it's not installed
try:
	import mlflow
	import mlflow.sklearn
	MLFLOW_AVAILABLE = True
except Exception:
	MLFLOW_AVAILABLE = False


def load_data(path: str) -> pd.DataFrame:
	if not os.path.exists(path):
		print(f"Error: data file not found: {path}", file=sys.stderr)
		sys.exit(2)

	df = pd.read_csv(path)
	return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
	categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
	numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

	# Build a OneHotEncoder in a way that's compatible with multiple
	# scikit-learn versions (some use `sparse`, newer ones use `sparse_output`).
	try:
		encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
	except TypeError:
		# Fallback for newer sklearn versions
		encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

	preprocessor = ColumnTransformer(
		transformers=[
			('num', 'passthrough', numerical_cols),
			('cat', encoder, categorical_cols),
		]
	)

	model = Pipeline(steps=[
		('preprocessor', preprocessor),
		('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
	])

	return model


def train_and_evaluate(model: Pipeline, X_train, X_test, y_train, y_test, show_progress: bool = True):
	"""Train the pipeline. If the classifier supports warm_start, train in chunks
	and show a tqdm progress bar (if available). Returns (accuracy, trained_pipeline).
	"""
	# Extract preprocessor and classifier
	preprocessor = model.named_steps['preprocessor']
	classifier = model.named_steps['classifier']

	# Fit/transform preprocessors once
	X_train_t = preprocessor.fit_transform(X_train)
	X_test_t = preprocessor.transform(X_test)

	total_estimators = getattr(classifier, 'n_estimators', None)
	supports_warm = hasattr(classifier, 'warm_start')

	if supports_warm and total_estimators and total_estimators > 1:
		# Train in chunks to provide progress updates
		chunk = max(1, total_estimators // 10)
		trained = 0

		if TQDM_AVAILABLE and show_progress:
			iterator = list(range(chunk, total_estimators + 1, chunk))
			if iterator[-1] != total_estimators:
				iterator.append(total_estimators)
			for n in tqdm(iterator, desc='Training trees', unit='trees'):
				classifier.warm_start = True
				classifier.n_estimators = n
				classifier.fit(X_train_t, y_train)
				trained = n
		else:
			print(f"Training RandomForest in chunks up to {total_estimators} trees...")
			for n in range(chunk, total_estimators + 1, chunk):
				classifier.warm_start = True
				classifier.n_estimators = n
				classifier.fit(X_train_t, y_train)
				trained = n
	else:
		# Fallback: single fit
		classifier.fit(X_train_t, y_train)

	# Build final pipeline with trained components
	trained_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])

	# Evaluate
	y_pred = trained_pipeline.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	return acc, trained_pipeline


def main():
	df = load_data(DATA_PATH)

	# Basic preprocessing
	df = df.drop(['customerID'], axis=1, errors='ignore')
	if 'TotalCharges' in df.columns:
		df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
	df = df.dropna()

	if 'Churn' not in df.columns:
		print('Error: target column "Churn" not found in data', file=sys.stderr)
		sys.exit(2)

	X = df.drop('Churn', axis=1)
	y = df['Churn'].map({'Yes': 1, 'No': 0})

	model = build_pipeline(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# If MLflow is available and a tracking URI was provided, attempt to log there.
	if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
		try:
			mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
			mlflow.set_experiment(MLFLOW_EXPERIMENT)
		except Exception as e:
			print(f'Warning: could not configure mlflow ({e}), continuing without mlflow logging')

	if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
		try:
			with mlflow.start_run():
				params = {'n_estimators': 100, 'random_state': 42}
				try:
					mlflow.log_params(params)
				except Exception:
					pass

				acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=True)
				print(f'Accuracy: {acc:.4f}')
				try:
					mlflow.log_metric('accuracy', float(acc))
				except Exception:
					pass

				try:
					if MLFLOW_REGISTER:
						mlflow.sklearn.log_model(model, 'model', registered_model_name=MLFLOW_REGISTERED_NAME)
					else:
						mlflow.sklearn.log_model(model, 'model')
				except Exception as e:
					print(f'Warning: failed to log/register model to MLflow: {e}')
		except Exception as e:
			print(f'Warning: mlflow run failed ({e}), training locally instead')
			acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=False)
			print(f'Accuracy (local): {acc:.4f}')
	else:
		# Train without MLflow logging
		acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=True)
		print(f'Accuracy (no mlflow): {acc:.4f}')

	# Always save local model
	os.makedirs(os.path.dirname(MODEL_PATH) or 'models', exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
	main()
