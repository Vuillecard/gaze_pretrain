import os 

path = '/idiap/resource/database/CCDb/'

L1 = []
for root, dirs, files in os.walk(os.path.join(path,'3D_no_annot')):
    for dir in dirs:
        
        for root2, dirs2, files2 in os.walk(os.path.join(path,'3D_no_annot', dir)):
            for file in files2:
                if file.endswith('.mov'):
                    print(file)
                    L1.append(file.split('.')[0])


L2 = []
for name in os.listdir(path):
    
    if name.startswith('P'):
        for root2, dirs2, files2 in os.walk(os.path.join(path, name)):
            for file in files2:
                if file.endswith('.mp4') or file.endswith('.mov'):
                    print(file)
                    L2.append(file.split('.')[0])
            

print(len(L1))
print(len(L2))
print(len(set(L2+L1)))

name_annotation = []
path_annot = '/idiap/project/epartners4all/data/datasets/datasets_final/ccdb/annotation'

for name in os.listdir(path_annot):
    if name.endswith('.csv'):
        name_annotation.append(name.replace('_annotation_matrix.csv',''))

print('annotaiont ',len(name_annotation))
print(set(L2+L1) - set(name_annotation))


L2_sub = []