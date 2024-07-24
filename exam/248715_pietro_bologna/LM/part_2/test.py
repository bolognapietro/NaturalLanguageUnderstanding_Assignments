import os

# Get the directory name of the file
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.train.txt")

# Print the directory
print(directory)
