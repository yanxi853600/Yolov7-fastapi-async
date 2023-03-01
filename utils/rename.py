import os

def batch_rename(path):
    count = 0
    for fname in os.listdir(path):
        new_fname = str(count)
        print (os.path.join(path, fname))
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))
        count = count + 1

batch_rename()