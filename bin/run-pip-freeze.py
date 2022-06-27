"""
Run `pip freeze` from python within Kaggle to get a copy of the dependencies
in the current environment
"""
import subprocess

bashCommand = "pip freeze"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

if output:
    print(output.decode())
if error:
    print("ERROR!")
    print(error.decode())
