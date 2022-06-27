I want to develop a good workflow against MNIST on Kaggle. This will help develop a generic workflow I can use against any competition. This can probably live on github, and maybe I can write it up into a Medium article

Some features it will include:
* Generate a single script by inserting contents of imported files
* Upload script to run on Kaggle Kernel and submit if successful, with git hash/comment to keep track code that gave a particular result
* Download logs/score automatically
* Use pytorch lightning as much as possible
* A config file that sets paths (conditional on whether we're local or on Kaggle), defines hyperparameters, and sets the dataset
* 3 datasets: micro - to confirm code runs, mini - to iterate quickly, full - whole thing, to run on kernel and submit
* automated tests, if possible

# TODO: address pytorch warnings
