from kaggle import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import mlflow

from mlflow.models import infer_signature


NUMERICAL_FEATURES = [
	"danceability",
	"loudness",
	"energy",
	"tempo",
	"valence",
	"speechiness",
	"liveness",
	"acousticness",
	"instrumentalness",
	"duration_ms",
	"year",
]

CATEGORICAL_FEATURES = [
	"genre",
]

TARGET = "verdict"
RANDOM_STATE = 42

MLFLOW_IDS = {
	"spotify_track_popularity": "994688289495687490"
}
MLFLOW_MODEL_URI = ""


def buildMlPipeline():
	# downloading dataset
	print('\ndownloading dataset...')
	api = KaggleApi()
	api.authenticate()
	api.dataset_download_files(
		dataset="amitanshjoshi/spotify-1million-tracks", path="./data", unzip=True
	)
	spotify_tracks = pd.read_csv("./data/spotify_data.csv")
	print(spotify_tracks.head())

	# add further dataset logging
	#dataset = mlflow.data.from_pandas(
	#	spotify_tracks,
	#	#source="amitanshjoshi/spotify-1million-tracks",
	#	name="Spotify Popularity Dataset",
	#	targets="popularity"
	#)
	#mlflow.log_input(dataset, context="training")

	# Add the popularity verdict
	print('\nadding popularity verdict...')
	spotify_tracks[TARGET] = spotify_tracks.apply(
		lambda row: 1 if row["popularity"] >= 50 else 0, axis=1
	)
	feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
	features = spotify_tracks[feature_columns + [TARGET]]
	print(features.head())

	print('\ncreating train/test data sets...')
	train_data, test_data = train_test_split(features, random_state=RANDOM_STATE)
	#train_data, test_data, y_train, y_test = train_test_split(features, feature_columns, random_state=RANDOM_STATE)
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	train_input = train_data[feature_columns]
	train_output = train_data[TARGET]
	train_input_ros, train_output_ros = RandomOverSampler(random_state=RANDOM_STATE).fit_resample(train_input, train_output)

	print('\ncreating pipelines...')
	numerical_pipeline = Pipeline([("encoder", StandardScaler())])
	categorical_pipeline = Pipeline([("encoder", OneHotEncoder())])

	preprocessing_pipeline = ColumnTransformer(
		[
			("numerical_preprocessor", numerical_pipeline, NUMERICAL_FEATURES),
			("categorical_pipeline", categorical_pipeline, CATEGORICAL_FEATURES),
		]
	)

	model = XGBClassifier(random_state=RANDOM_STATE)
	pipeline = Pipeline(
		[
			("preprocessor", preprocessing_pipeline),
			("estimator", model),
		]
	)

	pipeline.fit(train_input_ros, train_output_ros)

	#y_pred = model.predict(test_data)
	#signature = infer_signature(test_data, y_pred)

	# construct an evaluation dataset from the test set
	#eval_data = test_data
	#eval_data["target"] = y_test

	# add further model logging
	#mlflow.sklearn.log_model(
	#	sk_model=model,
	#	artifact_path="sklearn-model",
	#	signatire=signature,
	#	registered_model_name="XGBClassifier",
	#)
	MLFLOW_MODEL_URI = mlflow.get_artifact_uri("model")
	result = mlflow.evaluate(MLFLOW_MODEL_URI, test_data, targets='verdict',model_type="classifier",evaluators=["default"])



def main():
	print('\nstarting run...')


	print('\nsetting up mlFlow tracking...')
	mlflow.set_experiment(experiment_id=MLFLOW_IDS.get('spotify_track_popularity'))
	mlflow.set_tracking_uri('http://127.0.0.1:4000')
	mlflow.autolog(
		log_datasets=False,
		log_input_examples=True,
		log_model_signatures=True
	)

	with mlflow.start_run():
		mlflow.log_param("custom_logParam_01", 1)
		mlflow.log_metric("custom_logMetric_01", 0)

		buildMlPipeline()

	# not needed as with-block contains end_run()
	# mlflow.end_run(status='FINISHED')
	print('\nfinnished run...')




if __name__ == '__main__':
	main()
