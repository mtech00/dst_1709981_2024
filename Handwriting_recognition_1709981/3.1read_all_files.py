

import os
import subprocess

path = "./output"
dir_list = os.listdir(path)
count = 0

for file_name in dir_list:
    if file_name == ".DS_Store":
        continue  # Skip processing the ".DS_Store" file it is about macOS problem

    count += 1
    print(file_name, type(file_name))
    command = f"python 3.2conversion_picture_to_numbers.py {file_name}"
    subprocess.run(command, shell=True)

print("Files and directories in '", path, "' :")
print(dir_list)

# import os
# #import conversion_picture_to_numbers as cp
#
# path = "./output"
# dir_list = os.listdir(path)
# count=0
# dataset=""
#
# for x in dir_list:
#     count =count+1
#     print(x,type(x))
#     cp="python conversion_picture_to_numbers.py "+x
#     os.system(cp)
#
# print("Files and directories in '", path, "' :")
# print(dir_list)
