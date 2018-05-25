from sklearn.externals import joblib
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(working_dir, "saved")

'''
---------------------------------------------------------------------------------
README
---------------------------------------------------------------------------------
THIS IS A MODULAR SYSTEM THAT ALLOWS YOU TO SAVE/LOAD SKLEARN MODELS.
YOU MIGHT WANT TO SPECIFY FILE EXTENSIONS, BUT WE PREFER DEALING WITH IT SO JUST TELL ME THE FILE NAME WITHOUT EXTENSION
NOTE: IF YOU SPECIFY THE FILE EXTENSION THERE WON'T BE ANY PROBLEMS UNTIL YOU MAKE A TYPO
'''

# The higher, the slowest to read/write
compression_lv = 3

# Available : ‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’
compression_format = '.gz'


def add_extension(name, compression):
    extension = lambda c: '.pkl' if not c else compression_format
    return name if name.endswith(extension(compression)) else name + extension(compression)


# Save a model to the specified directory
# You can either pass the name with or without extension, we will check it anyway
# If compression is true the selected compression method will be used
# -----> For reference only, a decision tree without compression takes approximately 50MB,
# ---------------------> WITH COMPRESSION THE SAME TREE TAKES 9MB


def save_model(model, name, compression=True):
    # Check extension
    name = add_extension(name, compression)

    # Create file
    file = os.path.join(models_path, name)
    open(file, 'a').close()
    joblib.dump(model, file, compress=compression_lv if compression else 0)

# Load a model.
# Give me the model file name with or without the extension. I will load the right one anyway :)

def load_model(name):

    # Add extension (assume compression is on)
    f = add_extension(name, True)
    path = os.path.join(models_path, f)

    if not os.path.isfile(path):
        f = add_extension(name, False)
        path = os.path.join(models_path, f)

    return joblib.load(path)
