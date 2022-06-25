tests:
	coverage run -m pytest test --durations 20
	coverage report
