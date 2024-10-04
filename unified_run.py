
import os
import time
import subprocess
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "6,5,10,11,12,13"

models = {
    '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-5-CXRBERT_cls_token/epoch=122-train_loss= 4.43.ckpt'
    : [
        # ['fixed_each_unified', 1, 1],

        # ['fixed_each_unified', 1, 2],
        ['fixed_each_unified', 2, 2],

    #     ['fixed_each_unified', 1, 3],
    #     ['fixed_each_unified', 2, 3],
    #     ['fixed_each_unified', 3, 3],
    ]
}

test_meta_files = ['/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/test_main_file_v2.csv']

for model_path, configs in models.items():
    for config in configs:
        for ckpt in [model_path]:
            for meta_file in test_meta_files:
                EXP_PATH = os.getcwd()
                SRC_PATH = 'unified_main.py'
                TRAINING_CONFIG = {
                    'test': True,
                    'reload_ckpt_dir': ckpt,
                    'under_sample': config[0],
                    'max_img_num': config[1],
                    'target_count': config[2],
                    'test_meta_file': meta_file,
                    'n_gpus':6
                }
                TRAINING_CONFIG_LIST = list()
                for (k, v) in list(TRAINING_CONFIG.items()):
                    if (isinstance(v, bool)):
                        if v:
                            TRAINING_CONFIG_LIST.append("--{}={}".format(k, v))
                    else:
                        TRAINING_CONFIG_LIST.append("--{}={}".format(k, v))

                print('Training_lst:', TRAINING_CONFIG_LIST)
                subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)

                print('stopp')
                time.sleep(10)