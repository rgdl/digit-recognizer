# TODO: in time this might need to become it's own CLI
# TODO: but does that give any benefit over Make?
# TODO: can I use Make's feature of checking dependencies before rerunning?

tests:
	coverage run -m pytest test --durations 20
	coverage report -i

train-local:
	python src/train.py

analyse-errors:
	echo "analyse-errors not implemented yet"

tune-hyperparameters:
	# Use optuna like so: https://www.pytorchlightning.ai/blog/using-optuna-to-optimize-pytorch-lightning-hyperparameters
	echo "tune-hyperparameters not implemented yet"

generate-kaggle-script:
	python src/generate_kaggle_script.py src/train.py gen/main.py

run-in-kaggle:
	# Will involve a script-generation step
	make generate-kaggle-script
	echo "run-in-kaggle not implemented yet"

get-kaggle-results:
	echo "get-kaggle-results not implemented yet"
