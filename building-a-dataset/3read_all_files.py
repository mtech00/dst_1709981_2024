import os
#import conversion_picture_to_numbers as cp

path = "./output"
dir_list = os.listdir(path)
count=0
dataset=""

for x in dir_list:
    count =count+1
    print(x,type(x))
    cp="python conversion_picture_to_numbers.py "+x
    os.system(cp)

print("Files and directories in '", path, "' :")
print(dir_list)