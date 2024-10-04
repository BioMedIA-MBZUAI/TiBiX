import torch

file_name = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full.pt'
pt = torch.load(file_name)

print(len(pt))
print(len(pt[0]))
# breakpoint()
subject_ids = []
for i in range(len(pt)):
    subject_ids.extend(pt[i]['subject_ids'])

print(len((subject_ids)))
breakpoint()