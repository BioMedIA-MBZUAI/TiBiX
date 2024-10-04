import torch
import os
from nltk.translate.bleu_score import corpus_bleu
import csv
# /nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:0_test_output_epoch=122_1_of_1_test_main_file_v2.pt
# file_list = [
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:0_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:1_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:2_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:3_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:4_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/cuda:5_test_output_epoch=122_2_of_2_test_main_file_v2.pt',
# ]

# list_pt_files = []
# for path in file_list:
#     tmp = torch.load(path, map_location='cpu')
#     list_pt_files.extend(tmp)
# torch.save(list_pt_files, '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full_2_2.pt')

ckpt_path = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/full_2_2.pt'
# gathered_test_step_outputs = # GATHER FROM SOMEWHERE
# max_img_num = # DEFINE HERE
save_dir = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3/'
test_meta_file_name = 'test_main_file_v2'


gathered_test_step_outputs = torch.load(ckpt_path)
print(f"after gather, len = {len(gathered_test_step_outputs)}")
# print('test_epoch_After',len(gathered_test_step_outputs))
# print(gathered_test_step_outputs)
subs = []
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
cache_direc = "./biomed_VLP/"

from transformers import AutoModel, AutoTokenizer
from tokenizers.processors import BertProcessing
# tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, cache_dir = cache_direc)
tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[CLS]", "[SEP]", "[MASK]"]})


img_paths = gathered_test_step_outputs[0]['img_paths']
subject_id = gathered_test_step_outputs[0]['subject_ids']
max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['gen_text'])

for i, out in enumerate(gathered_test_step_outputs):
    GT_text = out['GT_text'].reshape(-1, max_text_len)
    gen_text = out['gen_text'].reshape(-1, max_text_len)
    sub = out['subject_ids']
    # print('subbbbbbbbbbbb', sub)
    subs.extend(sub)
    total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
    total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
GT_decoded_texts, gen_decoded_texts = [], []
for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
    gt_text_i = gt_text_i.tolist()
    gen_text_i = gen_text_i.tolist()
    # gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True, padding='max_length', max_length = 256, truncation = True)#tokenizer.decode(gt_text_i, skip_special_tokens=True)
    gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True)
    gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True)

    # gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True, padding='max_length', max_length = 256, truncation = True)#tokenizer.decode(gen_text_i, skip_special_tokens=True)
    GT_decoded_texts.append(gt_decoded_text_i)
    gen_decoded_texts.append(gen_decoded_text_i)
# calculate BLEU
references = []
candidates = []
for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
    reference = [gt_decoded_text_i.split(' ')]
    candidate = gen_decoded_text_i.split(' ')
    references.append(reference)
    candidates.append(candidate)

bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
print(f'Cumulative 1-gram: {bleu1:.3f}')
print(f'Cumulative 2-gram: {bleu2:.3f}')
print(f'Cumulative 3-gram: {bleu3:.3f}')
print(f'Cumulative 4-gram: {bleu4:.3f}')
print("test_BLEU-1", bleu1)
print("test_BLEU-2", bleu2)
print("test_BLEU-3", bleu3)
print("test_BLEU-4", bleu4)
# save csv files for labeler
GT_REPORTS_PATH = os.path.join(save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
    round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + ckpt_path.split('/')[-1].split('-')[0] + '_' + str(2) + '_of_' + str(2) + test_meta_file_name + '.csv')
GEN_REPORTS_PATH = os.path.join(save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
    round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + ckpt_path.split('/')[-1].split('-')[0] + '_' + str(2) + '_of_' + str(2) + test_meta_file_name + '.csv')
IMG_PATHS = os.path.join(save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(
                                round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + ckpt_path.split('/')[-1].split('-')[0] + '_' + str(2) + '_of_' + str(2) + test_meta_file_name + '.csv')
f_gt = open(GT_REPORTS_PATH, 'a')
wr_gt = csv.writer(f_gt)
f_gen = open(GEN_REPORTS_PATH, 'a')
wr_gen = csv.writer(f_gen)
f_img = open(IMG_PATHS, 'a')
wr_img = csv.writer(f_img)
# print('lennnnn', len(GT_decoded_texts), len(gen_decoded_texts), len(self.subs))
for gt_decoded_text_i, gen_decoded_text_i,subs_i in zip(GT_decoded_texts, gen_decoded_texts, subs):
    wr_gt.writerow([subs_i, gt_decoded_text_i])
    wr_gen.writerow([subs_i, gen_decoded_text_i])
for subs_i, img_paths_i in zip(img_paths, subs):
    wr_img.writerow([subs_i, img_paths_i])
f_gt.close()
f_gen.close()
f_img.close()
print("GEN_reports_test saved.")
print(f'\n\n')
