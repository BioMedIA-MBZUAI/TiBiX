import os
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from vae import VQGanVAE
from helpers import str2bool

os.environ["CUDA_VISIBLE_DEVICES"] = "12"


models = {
    # '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-4/test_output_epoch=109_1_of_1_test_main_file_v2.pt'
    #'/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full.pt'
    # '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-6-CXRBERT_wo_token/test_output_epoch=72_1_of_2_test_main_file_v2.pt',
    # '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full_2_2.pt'
    '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full_1_2.pt'
}

for infer_path in models:
    output_path = glob(os.path.join(infer_path))
    print(output_path)
    for output_pt_file in output_path:
        print('pt_file',output_pt_file)
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_save', default=True, type=str2bool, help='')
        parser.add_argument('--save_dir', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-1_results', type=str, help='')
        parser.add_argument('--infer_num', default=str(32), type=str, help='infer num when load eval ckpt')
        parser.add_argument('--vqgan_model_path', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/mimic_vqgan/last.ckpt', type=str)
        parser.add_argument('--vqgan_config_path', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/mimic_vqgan/2021-12-17T08-58-54-project.yaml', type=str)


        args = parser.parse_args()
        print(output_pt_file)
        args.save_dir = output_pt_file[:-3]
        print(args.save_dir)
        if args.img_save:
            os.makedirs(args.save_dir, exist_ok=True)

        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path).cuda()
        print(output_pt_file)
        output = torch.load(output_pt_file)
        for i, row in tqdm(enumerate(output), desc='len of file'):
            max_img_num = 0
            bsz = len(row['img_paths'])

            for k in row.keys():
                if k.startswith('GT_image'):
                    max_img_num += 1

            for b in tqdm(range(bsz), desc='bsz'):
                name_paths = row['img_paths'][b].split('|')[0].split('/')
                name = name_paths[-4] + "_" + row['subject_ids'][b] + "_" + name_paths[-3] + "_" + name_paths[-2]

                for i in range(1, max_img_num+1):
                    bsz, num_codes = row[f'GT_image{i}'].size()

                    GT_tensor = row[f'GT_image{i}'].reshape(-1, num_codes)[b][1:-1].unsqueeze(0)
                    gen_tensor = row[f'gen_image{i}'].reshape(-1, num_codes)[b][1:-1].unsqueeze(0)

                    GT_img1 = vae.decode(GT_tensor.to('cuda'))
                    torch.cuda.empty_cache()
                    gen_img1 = vae.decode(gen_tensor.to('cuda'))
                    torch.cuda.empty_cache()

                    if args.img_save:
                        GT_img1 = GT_img1[0].permute(1, 2, 0).detach().cpu().numpy()
                        gen_img1 = gen_img1[0].permute(1, 2, 0).detach().cpu().numpy()
                        plt.imsave(os.path.join(args.save_dir, name + f'_gen_img{i}.jpeg'), gen_img1)
                        plt.imsave(os.path.join(args.save_dir, name + f'_GT_img{i}.jpeg'), GT_img1)
                    del GT_img1
                    del gen_img1
                    torch.cuda.empty_cache()

        if args.img_save:
            print("\n >>> image saving done!")