# TODO: in time this might need to become it's own CLI
# TODO: but does that give any benefit over Make? Better handling of CL args for one
# TODO: can I use Make's feature of checking dependencies before rerunning?

tests:
	coverage run -m pytest test --durations 20 --profile
	coverage report -i

profile-results:
	python -c "import pstats; p = pstats.Stats('prof/combined.prof');  p.sort_stats('cumtime'); p.print_stats('digit-recognizer')"

train-local:
	python src/train.py

analyse-errors:
	python src/analyse_errors.py $(DIR)

tune-hyperparameters:
	# Use optuna like so: https://www.pytorchlightning.ai/blog/using-optuna-to-optimize-pytorch-lightning-hyperparameters
	echo "tune-hyperparameters not implemented yet"

run-in-kaggle:
	python src/generate_kaggle_script.py src/train.py gen/main.py
	bin/generate-kernel-metadata.sh
	kaggle kernels push -p gen

get-kaggle-results:
	bin/get-kaggle-results.sh

submit:
	kaggle competitions submit \
		-f data/output/kaggle_logs/submission.csv \
		-m b347239b309f0f6f911f7ccc0fb8e6536f8a4a91 \
		competition digit-recognizer
