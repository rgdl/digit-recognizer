# TODO: in time this might need to become it's own CLI
# TODO: but does that give any benefit over Make? Better handling of CL args for one

tests:
	coverage run -m pytest test --durations 20 --profile
	coverage report -i

profile-results:
	python -c "import pstats; p = pstats.Stats('prof/combined.prof');  p.sort_stats('cumtime'); p.print_stats('digit-recognizer')"

tune-local:
	python src/tune.py

tune-kaggle:
	python src/generate_kaggle_script.py src/tune.py gen/main.py
	bin/generate-kernel-metadata.sh
	kaggle kernels push -p gen

train-local:
	python src/train.py

train-kaggle:
	python src/generate_kaggle_script.py src/train.py gen/main.py
	bin/generate-kernel-metadata.sh
	kaggle kernels push -p gen

analyse-errors:
	# TODO: Local train results only - kaggle version too?
	python src/analyse_errors.py data/output/evaluation

analyse-tuning:
	python src/analyse_tuning.py data/output/kaggle_logs/tune.pickle

get-kaggle-results:
	bin/get-kaggle-results.sh

submit:
	kaggle competitions submit \
		-f data/output/kaggle_logs/submission.csv \
		-m $(git rev-parse HEAD) \
		digit-recognizer
