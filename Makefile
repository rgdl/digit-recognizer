# TODO: in time this might need to become it's own CLI
# TODO: but does that give any benefit over Make?

tests:
	coverage run -m pytest test --durations 20
	coverage report -i

train-local:
	python src/train.py

analyse-errors:
	echo "analyse-errors not implemented yet"

tune-hyperparameters:
	echo "tune-hyperparameters not implemented yet"

run-in-kaggle:
	echo "run-in-kaggle not implemented yet"

get-kaggle-results:
	echo "get-kaggle-results not implemented yet"
