import os
import csv
import math
import time
import random
import numpy as np
import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn.metrics
import sklearn, sklearn.model_selection
import torch.distributed as dist
import pytorch_lightning as pl

from nltk.translate.bleu_score import corpus_bleu
from transformer_pytorch.transformer_unified import TransformerLM_unified
from transformers.optimization import get_cosine_schedule_with_warmup

random.seed(42)
import torch.distributed as dist
## in lightning module

cache_direc = "./biomed_VLP/"

# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"

class TransformerLightning_unified(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01,
                 pad_token_idx=0, sos_token_idx=2, eos_token_idx=3,#sansan
                 save_dir="", causal_trans='conditioned_causal', **kargs):
        super().__init__()
        self.kargs = kargs
        self.max_img_num = kargs['max_img_num']
        self.under_sample = kargs['under_sample']
        self.attn_type = kargs['attn_type']
        self.num_txt_tokens = kargs['num_tokens']
        self.num_img_tokens = kargs['num_img_tokens']
        self.taskweights = kargs['weights']

        self.ckpt_path = None
        self.target_count = None
        self.test_meta_file_name = None

        self.transformerLM_unified = TransformerLM_unified(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir#'/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/exp-6_debug_v3'#'/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp_6_debug_v3/'#'/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-5-CXRBERT_cls_token' #save_dir 
        self.causal = causal_trans
        self.subs = []

        self.save_hyperparameters(ignore=['tokenizer'])

        self.task_outputs={}
        self.task_targets={}
        
        for task in range(13):
            self.task_outputs[task] = []
            self.task_targets[task] = []
        # ADAM
        self.sigmoid = nn.Sigmoid()
        self.cls_criterion = torch.nn.BCEWithLogitsLoss()

    def merge(self,outputs):
        if dist.is_initialized():
            all_rank_outputs = [None for _ in range(dist.get_world_size())]    
            dist.all_gather_object(all_rank_outputs,outputs)
            outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret

    def classification_metric_evaluation(self, auc, task_aucs, task_outputs, task_targets):
        perf_dict = {}
        all_threshs = []#[0.58074194, 0.54576606, np.nan, 0.5225241, 0.66747427, np.nan, np.nan, 0.7036588, 0.521654, np.nan, 0.5775543, np.nan, np.nan, np.nan, 0.57897043, 0.520638, 0.6373577, 0.5326836]
        all_min = []
        all_max = []
        all_ppv80 = []
        all_accuracy = []
        all_f1_score = []
        all_precision = []
        all_recall = []
        all_auc = []
        results = [auc, task_aucs, task_outputs, task_targets]
        pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum', 'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'Pleural_Effusion', 'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
        for i, patho in enumerate(pathologies):
            # print(i, patho)
            opt_thres = np.nan
            opt_min = np.nan
            opt_max = np.nan
            ppv80_thres = np.nan
            accuracy = np.nan
            f1_score = np.nan
            precision = np.nan
            recall = np.nan
            auc = np.nan
            
            if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
                #sigmoid
                # all_outputs =  1.0/(1.0 + np.exp(-results[2][i]))
                all_outputs = results[2][i]
                fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
                pente = tpr - fpr
                opt_thres = thres_roc[np.argmax(pente)]
                opt_min = all_outputs.min()
                opt_max = all_outputs.max()
                
                ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
                ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
                ppv80_thres = thres_pr[ppv80_thres_idx-1]
                
                auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
                
                # Calculate confusion matrix for accuracy, precision, recall, and F1 score
                threshold = opt_thres #all_threshs[i]  
                predicted_labels = (all_outputs >= threshold).astype(int)
                true_labels = results[3][i]
                confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
                TP = confusion_matrix[1, 1]
                TN = confusion_matrix[0, 0]
                FP = confusion_matrix[0, 1]
                FN = confusion_matrix[1, 0]

                # Calculate metrics
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                # # Add metrics to perf_dict
                # perf_dict[patho] = {
                #     'AUC': round(auc, 2),
                #     'Accuracy': round(accuracy, 2),
                #     'F1 Score': round(f1_score, 2),
                #     'Precision': round(precision, 2),
                #     'Recall': round(recall, 2)
                # }
                
                all_auc.append(auc)  # Append AUC to the list
                
            else:
                perf_dict[patho] = "-"
            
            # Append metrics to respective lists
            all_threshs.append(opt_thres)
            all_min.append(opt_min)
            all_max.append(opt_max)
            all_ppv80.append(ppv80_thres)
            all_accuracy.append(accuracy)
            all_f1_score.append(f1_score)
            all_precision.append(precision)
            all_recall.append(recall)

        # Print the results
        # print("pathologies", pathologies)
        # print("------------------------------------------------------------------------------------------------")
        # print("op_threshs", str(all_threshs).replace("nan", "np.nan"))
        # print("min", str(all_min).replace("nan", "np.nan"))
        # print("max", str(all_max).replace("nan", "np.nan"))
        # print("ppv80", str(all_ppv80).replace("nan", "np.nan"))
        # print("accuracy", str(all_accuracy).replace("nan", "np.nan"))
        # print("f1_score", str(all_f1_score).replace("nan", "np.nan"))
        # print("precision", str(all_precision).replace("nan", "np.nan"))
        # print("recall", str(all_recall).replace("nan", "np.nan"))
        # print("all AUC values:", str(all_auc).replace("nan", "np.nan"))

        # Calculate and print average metrics
        avg_accuracy = np.nanmean(all_accuracy)
        avg_f1_score = np.nanmean(all_f1_score)
        avg_precision = np.nanmean(all_precision)
        avg_recall = np.nanmean(all_recall)
        avg_auc = np.nanmean(all_auc)

        # print(f'Average Accuracy: {round(avg_accuracy, 2)}')
        # print(f'Average F1 Score: {round(avg_f1_score, 2)}')
        # print(f'Average Precision: {round(avg_precision, 2)}')
        # print(f'Average Recall: {round(avg_recall, 2)}')
        # print(f'Average AUC: {round(avg_auc, 2)}')
        print(f'Average Accuracy: {round(avg_accuracy, 2)}, Average F1 Score: {round(avg_f1_score, 2)}, Average Precision: {round(avg_precision, 2)}, Average Recall: {round(avg_recall, 2)}, Average AUC: {round(avg_auc, 2)}')
        self.log('Average Accuracy:', round(avg_accuracy, 2), on_step=False, on_epoch=True, sync_dist=True)
        self.log('Average F1 Score:', round(avg_f1_score, 2), on_step=False, on_epoch=True, sync_dist=True)
        self.log('Average Precision:', round(avg_precision, 2), on_step=False, on_epoch=True, sync_dist=True)
        self.log('Average Recall:', round(avg_recall, 2), on_step=False, on_epoch=True, sync_dist=True)
        self.log('Average AUC:', round(avg_auc, 2), on_step=False, on_epoch=True, sync_dist=True)


        return avg_accuracy, avg_f1_score, avg_precision, avg_recall, avg_auc

    def forward(self, batch):
        # logit = self.transformerLM_unified(batch, causal=self.causal)
        # return logit
        # Adam
        logit, cls_logit = self.transformerLM_unified(batch, causal=self.causal)
        return logit, cls_logit

    def training_step(self, batch, batch_idx):
        img1, txt, modes, view, img_state, cls_targets, weights = batch['img1'], batch['txt'], batch['modes'], batch['view_position'], batch['image_state'], batch['labels'],batch['weights']
        # print(cls_targets, weights)
        assert txt.shape[0] == img1.shape[0]
        batch_size = txt.shape[0]
        txt_seq_len = txt.shape[1]
        img_seq_len = img1.shape[1]
        n = img_seq_len + txt_seq_len
        if 'img2' in batch.keys():
            img2 = batch['img2']
            n += img2.shape[1]
        if 'img3' in batch.keys():
            img3 = batch['img3']
            n += img3.shape[1]

        # logit = self(batch)[:, :-1, :]
        # ADAM
        logit, cls_logit = self(batch)
        logit = logit[:, :-1, :]
        max_neg_value = -torch.finfo(logit.dtype).max

        for bsz in range(batch_size):
            if np.array(modes)[:, bsz][0] == 'txt':
                first_modal = txt_seq_len - 1
                logit[bsz, :first_modal, self.num_txt_tokens:] = max_neg_value
                logit[bsz, first_modal:, :self.num_txt_tokens] = max_neg_value
            else:
                first_modal = img_seq_len - 1
                if np.array(modes)[:, bsz][1] == 'txt':
                    logit[bsz, :first_modal, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, first_modal: (first_modal + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, (first_modal + txt_seq_len):, :self.num_txt_tokens] = max_neg_value
                elif np.array(modes)[:, bsz][-1] == 'txt':
                    logit[bsz, :-txt_seq_len, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, -txt_seq_len:, self.num_txt_tokens:] = max_neg_value
                if 'img3' in batch.keys() and np.array(modes)[:, bsz][2] == 'txt':  # [i, i, t, i]
                    logit[bsz, :(first_modal + img_seq_len), :self.num_txt_tokens] = max_neg_value
                    logit[bsz, (first_modal + img_seq_len):(first_modal + img_seq_len + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, -img_seq_len:, :self.num_txt_tokens] = max_neg_value

        logit = logit.reshape(-1, logit.size(-1))

        target_lst = []
        for bsz in range(batch_size):
            for idx, mode in enumerate(np.array(modes)[:, bsz]):
                if idx == 0:
                    target = batch[mode][bsz, 1:]
                else:
                    target = batch[mode][bsz]
                if mode.startswith('img'):
                    target_lst.append(target + self.num_txt_tokens)
                else:
                    target_lst.append(target)
        target = torch.cat(target_lst, dim=0)

        ignore_classes = torch.ones(self.num_txt_tokens + self.num_img_tokens)
        ignore_classes[1024 + self.num_txt_tokens] = 0.
        gen_loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx, weight=ignore_classes.to(logit.device))
        weights = (weights).to('cuda').float()
        # print('logitttttttttt', cls_logit)

        # ADAM
        cls_loss = torch.zeros(1).to('cuda').float()
        for task in range(cls_targets.shape[1]):
            task_output = cls_logit[:,task]
            task_target = cls_targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = self.cls_criterion(task_output.float(), task_target.float())
                if self.taskweights:
                    cls_loss += weights[0][task]*task_loss
                else:
                    cls_loss += task_loss
        loss = gen_loss + 0.1 * cls_loss
        # print(loss,task_loss, loss-task_loss)

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_gen_loss', gen_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_cls_loss', cls_loss, on_step=True, on_epoch=True, sync_dist=True)

        output = { #sansan
            'batch_idx': batch_idx,
            'loss': loss
        }
        return output

    def validation_step(self, batch, batch_idx):
        img1, txt, modes, view, img_state, cls_targets, weights = batch['img1'], batch['txt'], batch['modes'], batch['view_position'], batch['image_state'], batch['labels'],batch['weights']
        # print(cls_targets, weights)
        assert txt.shape[0] == img1.shape[0]
        batch_size = txt.shape[0]
        txt_seq_len = txt.shape[1]
        img_seq_len = img1.shape[1]
        n = img_seq_len + txt_seq_len
        if 'img2' in batch.keys():
            img2 = batch['img2']
            n += img2.shape[1]
        if 'img3' in batch.keys():
            img3 = batch['img3']
            n += img3.shape[1]
        # logit = self(batch)[:, :-1, :]
        # ADAM
        logit, cls_logit = self(batch)
        logit = logit[:, :-1, :]
        max_neg_value = -torch.finfo(logit.dtype).max
        for bsz in range(batch_size):
            if np.array(modes)[:, bsz][0] == 'txt':
                first_modal = txt_seq_len - 1
                logit[bsz, :first_modal, self.num_txt_tokens:] = max_neg_value
                logit[bsz, first_modal:, :self.num_txt_tokens] = max_neg_value
            else:
                first_modal = img_seq_len - 1
                if np.array(modes)[:, bsz][1] == 'txt':
                    logit[bsz, :first_modal, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, first_modal: (first_modal + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, (first_modal + txt_seq_len):, :self.num_txt_tokens] = max_neg_value
                elif np.array(modes)[:, bsz][-1] == 'txt':
                    logit[bsz, :-txt_seq_len, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, -txt_seq_len:, self.num_txt_tokens:] = max_neg_value
                if 'img3' in batch.keys() and np.array(modes)[:, bsz][2] == 'txt':  # [i, i, t, i]
                    logit[bsz, :(first_modal + img_seq_len), :self.num_txt_tokens] = max_neg_value
                    logit[bsz, (first_modal + img_seq_len):(first_modal + img_seq_len + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, -img_seq_len:, :self.num_txt_tokens] = max_neg_value

        logit = logit.reshape(-1, logit.size(-1))

        target_lst = []
        for bsz in range(batch_size):
            for idx, mode in enumerate(np.array(modes)[:, bsz]):
                if idx == 0:
                    target = batch[mode][bsz, 1:]
                else:
                    target = batch[mode][bsz]
                if mode.startswith('img'):
                    target_lst.append(target + self.num_txt_tokens)
                else:
                    target_lst.append(target)
        target = torch.cat(target_lst, dim=0)

        ignore_classes = torch.ones(self.num_txt_tokens + self.num_img_tokens)
        ignore_classes[1024 + self.num_txt_tokens] = 0.
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx, weight=ignore_classes.to(logit.device))
        gen_loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx, weight=ignore_classes.to(logit.device))
        weights = (weights).to('cuda').float()
        # print('weightsss',weights)

        # ADAM
        cls_loss = torch.zeros(1).to('cuda').float()
        for task in range(cls_targets.shape[1]):
            task_output = cls_logit[:,task]
            task_target = cls_targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                cls_loss = self.cls_criterion(task_output.float(), task_target.float()) #sansan
                # if self.taskweights:
                #     cls_loss += weights[0][task]*task_loss
                # else:
                #     cls_loss += task_loss
            # print(type(self.task_outputs[task]), type(self.task_targets[task]))
            self.task_outputs[task].append(self.sigmoid(task_output).detach().cpu().numpy()) #sansan
            self.task_targets[task].append(task_target.detach().cpu().numpy()) #sansan
            
            
        loss = gen_loss + 0.1 * cls_loss
        # print(loss,task_loss, loss-task_loss)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_gen_loss', gen_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_cls_loss', cls_loss, on_step=True, on_epoch=True, sync_dist=True)

        # output = {
        #     'outputs': cls_output_list, #sansan
        #     'targets': cls_target_list, #sansan
        # }
        # return output
    
    def validation_epoch_end(self, outputs):

        for task in range(len(self.task_targets)):
            self.task_outputs[task] = np.concatenate(self.task_outputs[task])
            self.task_targets[task] = np.concatenate(self.task_targets[task])
    
        task_aucs = []
        for task in range(len(self.task_targets)):
            if len(np.unique(self.task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(self.task_targets[task], self.task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

        task_aucs = np.asarray(task_aucs)
        auc = np.mean(task_aucs[~np.isnan(task_aucs)])
        self.log('AUC', auc, on_step=False, on_epoch=True, sync_dist=True)
        avg_accuracy, avg_f1_score, avg_precision, avg_recall, avg_auc = self.classification_metric_evaluation(auc, task_aucs, self.task_outputs, self.task_targets)
        

        self.task_outputs={}
        self.task_targets={}
        
        for task in range(13):
            self.task_outputs[task] = []
            self.task_targets[task] = []
        
        torch.cuda.empty_cache()


    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        dicom_id, img_paths, subject_ids = batch['dicom_id'], batch['img_paths'], batch['subject_id']
        img1, txt, modes, view, img_state = batch['img1'], batch['txt'], batch['modes'], batch['view_position'], batch['image_state']
        n = img1.shape[1] + txt.shape[1]
        self.transformerLM_unified.max_img_num = self.max_img_num

        if self.max_img_num == 1:
            modes_txt = [['img1'], ['txt']]
            modes_img1 = [['txt'], ['img1']]

        elif self.max_img_num == 2:
            n += batch['img2'].shape[1]

            modes_txt = random.sample([[['img1'], ['img2'], ['txt']], [['img2'], ['img1'], ['txt']]], 1)[0]
            modes_img1 = random.sample([[['img2'], ['txt'], ['img1']], [['txt'], ['img2'], ['img1']]], 1)[0]
            modes_img2 = random.sample([[['img1'], ['txt'], ['img2']], [['txt'], ['img1'], ['img2']]], 1)[0]

        elif self.max_img_num == 3:
            n += (batch['img2'].shape[1] + batch['img3'].shape[1])

            modes_txt = random.sample([['img1'], ['img2'], ['img3']], 3)
            modes_txt.append(['txt'])
            modes_img1 = random.sample([['txt'], ['img2'], ['img3']], 3)
            modes_img1.append(['img1'])
            modes_img2 = random.sample([['img1'], ['txt'], ['img3']], 3)
            modes_img2.append(['img2'])
            modes_img3 = random.sample([['img1'], ['img2'], ['txt']], 3)
            modes_img3.append(['img3'])

        # generate texts
        print('-'*30)
        print('generate txt')
        batch['modes'] = modes_txt

        gen_texts = self.transformerLM_unified.generate_texts(
            batch,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # generate img1
        print('-'*30)
        print('generate img1')
        batch['modes'] = modes_img1
        import copy
        tmp_batch_view = copy.deepcopy(batch['view_position'])
        batch['view_position'][-1], batch['view_position'][0] = batch['view_position'][0], batch['view_position'][-1]
        tmp_batch_image_state = copy.deepcopy(batch['image_state'])
        batch['image_state'][-1], batch['image_state'][0] = batch['image_state'][0], batch['image_state'][-1]

        gen_images1 = self.transformerLM_unified.generate_image(
            batch,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )
        batch['view_position'] = copy.deepcopy(tmp_batch_view)
        batch['image_state']   = copy.deepcopy(tmp_batch_image_state)


        if 'img2' in batch.keys():
            # generate img2
            batch['modes'] = modes_img2
            import copy
            tmp_batch_view = copy.deepcopy(batch['view_position'])
            batch['view_position'][-1], batch['view_position'][1] = batch['view_position'][1], batch['view_position'][-1]
            tmp_batch_image_state = copy.deepcopy(batch['image_state'])
            batch['image_state'][-1], batch['image_state'][1] = batch['image_state'][1], batch['image_state'][-1]

            gen_images2 = self.transformerLM_unified.generate_image(
                batch,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )
            batch['view_position'] = copy.deepcopy(tmp_batch_view)
            batch['image_state']   = copy.deepcopy(tmp_batch_image_state)

        if 'img3' in batch.keys():
            raise ValueError("HAHAHAHAHAHA")
            # generate img3
            batch['modes'] = modes_img3
            # import copy
            # tmp_batch_view = copy.deepcopy(batch['view_position'])
            # batch['view_position'][-1] = batch['view_position'][2]

            gen_images3 = self.transformerLM_unified.generate_image(
                batch,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )
            # batch['view_position'] = copy.deepcopy(tmp_batch_view)

        output = {
            'subject_ids':subject_ids,
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': img1,
            'gen_image1': gen_images1,
            'img_paths': img_paths,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'view': view,
            'img_state': img_state,
        }

        if 'img2' in batch.keys():
            output['GT_image2'] = batch['img2']
            output['gen_image2'] = gen_images2
            output['modes_img2'] = modes_img2

        if 'img3' in batch.keys():
            output['GT_image3'] = batch['img3']
            output['gen_image3'] = gen_images3
            output['modes_img3'] = modes_img3
        # After generating gen_texts
        # print(f'gen_texts on GPU {self.device}:', gen_texts)

        # # After generating gen_images1
        # print(f'gen_images1 on GPU {self.device}:', gen_images1)
        # print(out)
        # gathered_outputs = self.all_gather(output)
        # print('gathered outputsssssss test_epoch_step',len(gathered_outputs))
        print(len(output))
        return output

    def test_epoch_end(self, test_step_outputs):
        # from tokenizers import ByteLevelBPETokenizer
        # # from transformers import AutoModel, AutoTokenizer
        # from tokenizers.processors import BertProcessing
        # tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        # # tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, cache_dir = cache_direc)
        # tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        # tokenizer._tokenizer.post_processor = BertProcessing(
        #     ("[EOS]", tokenizer.token_to_id("[EOS]")),
        #     ("[SOS]", tokenizer.token_to_id("[SOS]")),
        # )
        # tokenizer.enable_truncation(max_length=256)
        # tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=256)

        # from tokenizers import ByteLevelBPETokenizer
        from transformers import AutoModel, AutoTokenizer
        from tokenizers.processors import BertProcessing
        # tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, cache_dir = cache_direc)
        tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[CLS]", "[SEP]", "[MASK]"]})

        print('test_epoch_end start',len(test_step_outputs))
        torch.save(test_step_outputs, os.path.join(self.save_dir, f"{self.device}_test_output_{self.ckpt_path.split('/')[-1].split('-')[0]}_{str(self.max_img_num)}_of_{str(self.target_count)}_{self.test_meta_file_name}.pt"))
        
        # if self.global_rank == 0:
        #     gathered_test_step_outputs = self.all_gather(test_step_outputs)
        #     print(f"after gather, len = {len(gathered_test_step_outputs)}")
        #     # print('test_epoch_After',len(gathered_test_step_outputs))
        #     # print(gathered_test_step_outputs)
        #     img_paths = gathered_test_step_outputs[0]['img_paths']
        #     subject_id = gathered_test_step_outputs[0]['subject_ids']
        #     if self.max_img_num != -1:
        #         max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
        #         total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
        #         total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['gen_text'])

        #         for i, out in enumerate(gathered_test_step_outputs):
        #             GT_text = out['GT_text'].reshape(-1, max_text_len)
        #             gen_text = out['gen_text'].reshape(-1, max_text_len)
        #             sub = out['subject_ids']
        #             # print('subbbbbbbbbbbb', sub)
        #             self.subs.extend(sub)
        #             total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
        #             total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)
        #     print('hiiiiiiiii')
        #     if self.max_img_num != -1:
        #         torch.save(gathered_test_step_outputs, os.path.join(self.save_dir, f"test_output_{self.ckpt_path.split('/')[-1].split('-')[0]}_{str(self.max_img_num)}_of_{str(self.target_count)}_{self.test_meta_file_name}_v2.pt"))
        #         # !# For generated texts
        #         GT_decoded_texts, gen_decoded_texts = [], []
        #         for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
        #             gt_text_i = gt_text_i.tolist()
        #             gen_text_i = gen_text_i.tolist()
        #             # gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True, padding='max_length', max_length = 256, truncation = True)#tokenizer.decode(gt_text_i, skip_special_tokens=True)
        #             gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True)
        #             gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True)

        #             # gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True, padding='max_length', max_length = 256, truncation = True)#tokenizer.decode(gen_text_i, skip_special_tokens=True)
        #             GT_decoded_texts.append(gt_decoded_text_i)
        #             gen_decoded_texts.append(gen_decoded_text_i)
        #         # calculate BLEU
        #         references = []
        #         candidates = []
        #         for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
        #             reference = [gt_decoded_text_i.split(' ')]
        #             candidate = gen_decoded_text_i.split(' ')
        #             references.append(reference)
        #             candidates.append(candidate)

        #         bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
        #         bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
        #         bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
        #         bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
        #         print(f'Cumulative 1-gram: {bleu1:.3f}')
        #         print(f'Cumulative 2-gram: {bleu2:.3f}')
        #         print(f'Cumulative 3-gram: {bleu3:.3f}')
        #         print(f'Cumulative 4-gram: {bleu4:.3f}')
        #         self.log("test_BLEU-1", bleu1)
        #         self.log("test_BLEU-2", bleu2)
        #         self.log("test_BLEU-3", bleu3)
        #         self.log("test_BLEU-4", bleu4)
        #         # save csv files for labeler
        #         GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
        #             round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
        #         GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
        #             round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
        #         IMG_PATHS = os.path.join(self.save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(
        #                                      round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
        #         f_gt = open(GT_REPORTS_PATH, 'a')
        #         wr_gt = csv.writer(f_gt)
        #         f_gen = open(GEN_REPORTS_PATH, 'a')
        #         wr_gen = csv.writer(f_gen)
        #         f_img = open(IMG_PATHS, 'a')
        #         wr_img = csv.writer(f_img)
        #         # print('lennnnn', len(GT_decoded_texts), len(gen_decoded_texts), len(self.subs))
        #         for gt_decoded_text_i, gen_decoded_text_i,subs_i in zip(GT_decoded_texts, gen_decoded_texts, self.subs):
        #             wr_gt.writerow([subs_i, gt_decoded_text_i])
        #             wr_gen.writerow([subs_i, gen_decoded_text_i])
        #         for subs_i, img_paths_i in zip(img_paths, self.subs):
        #             wr_img.writerow([subs_i, img_paths_i])
        #         f_gt.close()
        #         f_gen.close()
        #         f_img.close()
        #         print("GEN_reports_test saved.")
        #         print(f'\n\n')
        #         # print('hiiiiiii')
        #     # print('hellllloooo')
        # # print('endddddddddd')
        # time.sleep(0.5)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        train_loader = self.train_dataloader()
        scheduler = {
            'scheduler':
                get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=self.kargs['epochs'] * len(train_loader)),
            'interval': 'step',
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}    # def test_epoch_end(self, test_step_outputs):
    #     from tokenizers import ByteLevelBPETokenizer
    #     from tokenizers.processors import BertProcessing
    #     tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
    #     tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
    #     tokenizer._tokenizer.post_processor = BertProcessing(
    #         ("[EOS]", tokenizer.token_to_id("[EOS]")),
    #         ("[SOS]", tokenizer.token_to_id("[SOS]")),
    #     )
    #     tokenizer.enable_truncation(max_length=256)
    #     tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=256)


    #     gathered_test_step_outputs = self.all_gather(test_step_outputs)

    #     img_paths = gathered_test_step_outputs[0]['img_paths']
    #     if self.max_img_num != -1:
    #         max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
    #         total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
    #         total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])

    #         for i, out in enumerate(gathered_test_step_outputs):
    #             GT_text = out['GT_text'].reshape(-1, max_text_len)
    #             gen_text = out['gen_text'].reshape(-1, max_text_len)
    #             total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
    #             total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)

    #     if self.global_rank == 0:
    #         if self.max_img_num != -1:
    #             torch.save(gathered_test_step_outputs, os.path.join(self.save_dir, f"test_output_{self.ckpt_path.split('/')[-1].split('-')[0]}_{str(self.max_img_num)}_of_{str(self.target_count)}_{self.test_meta_file_name}.pt"))
    #             # !# For generated texts
    #             GT_decoded_texts, gen_decoded_texts = [], []
    #             for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
    #                 gt_text_i = gt_text_i.tolist()
    #                 gen_text_i = gen_text_i.tolist()
    #                 gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True)
    #                 gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True)
    #                 GT_decoded_texts.append(gt_decoded_text_i)
    #                 gen_decoded_texts.append(gen_decoded_text_i)
    #             # calculate BLEU
    #             references = []
    #             candidates = []
    #             for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
    #                 reference = [gt_decoded_text_i.split(' ')]
    #                 candidate = gen_decoded_text_i.split(' ')
    #                 references.append(reference)
    #                 candidates.append(candidate)

    #             bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    #             bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
    #             bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
    #             bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
    #             print(f'Cumulative 1-gram: {bleu1:.3f}')
    #             print(f'Cumulative 2-gram: {bleu2:.3f}')
    #             print(f'Cumulative 3-gram: {bleu3:.3f}')
    #             print(f'Cumulative 4-gram: {bleu4:.3f}')
    #             self.log("test_BLEU-1", bleu1)
    #             self.log("test_BLEU-2", bleu2)
    #             self.log("test_BLEU-3", bleu3)
    #             self.log("test_BLEU-4", bleu4)
    #             # save csv files for labeler
    #             GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
    #                 round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
    #             GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
    #                 round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
    #             IMG_PATHS = os.path.join(self.save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(
    #                                          round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
    #             f_gt = open(GT_REPORTS_PATH, 'a')
    #             wr_gt = csv.writer(f_gt)
    #             f_gen = open(GEN_REPORTS_PATH, 'a')
    #             wr_gen = csv.writer(f_gen)
    #             f_img = open(IMG_PATHS, 'a')
    #             wr_img = csv.writer(f_img)
    #             for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
    #                 wr_gt.writerow([gt_decoded_text_i])
    #                 wr_gen.writerow([gen_decoded_text_i])
    #             for img_paths_i in img_paths:
    #                 wr_img.writerow([img_paths_i])
    #             f_gt.close()
    #             f_gen.close()
    #             f_img.close()
    #             print("GEN_reports_test saved.")
    #             print(f'\n\n')

    #     time.sleep(0.5)

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.lr)
    #     train_loader = self.train_dataloader()
    #     scheduler = {
    #         'scheduler':
    #             get_cosine_schedule_with_warmup(
    #                 optimizer=optimizer,
    #                 num_warmup_steps=0,
    #                 num_training_steps=self.kargs['epochs'] * len(train_loader)),
    #         'interval': 'step',
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}