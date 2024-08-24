from email.mime import image
import os 
from time import time
import shutil
from multiprocessing.pool import ThreadPool

try:
    tmp_dir = os.environ["TMPDIR"]
    data_location = tmp_dir
except:
    data_location = None

dir_data = '/idiap/project/epartners4all/data/uniface_database/'

# start = time()
# os.system(f"cp -r {dir_data} {data_location}")
# print(f"Time to copy data: {time()-start}")

# List all files in dir_data

#print(file_list)

# Copy the entire directory from dir_data to data_location
# start = time()
# data_location = os.path.join(data_location, 'Gaze360_bis2')
# shutil.copytree(dir_data, data_location)
# end = time()
# print(f"Time to copy images: {end-start}")

dir_data = '/idiap/project/epartners4all/data/uniface_database/'

def _copy_image_to_tmp(image_path):
    dst_tmp = image_path.replace(dir_data, '')
    dst = os.path.join(data_location, dst_tmp)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(image_path, dst)

datasets_names = ['Gaze360' , 'Gazefollow' , 'GFIE' , 'MPSGaze']

print("collecting images to copy")
file_list = []
for dataset_name in datasets_names:
    for root, dirs, files in os.walk(os.path.join(dir_data, dataset_name)):
        for file in files:
            if file.endswith("head_crop.jpg"):
                file_list.append(os.path.join(root, file))

print("start copying images")
start = time()
total_imgs = len(file_list)
print(f"Total images: {total_imgs}")
with ThreadPool(8) as p:
    p.map(_copy_image_to_tmp, file_list)
end = time()
print(f"Time to copy images: {end-start}")