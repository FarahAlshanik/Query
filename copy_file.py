import shutil, os
files = ['file1.txt', 'file2.txt', 'file3.txt']
for f in files:
    shutil.copy(f, 'dest_folder')
