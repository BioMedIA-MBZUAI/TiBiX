import os
import csv
import random
import pickle
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import albumentations
import albumentations.pytorch
import re
import torch
from torch.utils.data import Dataset
import numpy as np
from vae import VQGanVAE
from datetime import datetime

random.seed(42)

#########################################
# ADAM TEMPORAL
from datetime import date

def rem_time(d):
    s = date(d.year,d.month, d.day)
    return s
#########################################

def datatime_conversion(studydate,studytime):
    studytime = str(studytime)
    studytime = studytime.split('.')[0]
    studytime_str = str(int(studytime)).zfill(6)
    studytime_str = studytime_str[:2] + ':' + studytime_str[2:4] + ':' + studytime_str[4:]
    studydatetime_str = str(studydate) + ' ' + studytime_str
    # print(studydatetime_str)
    studydatetime = datetime.strptime(studydatetime_str, '%Y%m%d %H:%M:%S')

    return studydatetime


def difference(prevtimestamp, currtimestamp):
    difference = currtimestamp - prevtimestamp
    years = difference.days // 365
    remaining_days = difference.days % 365
    months = remaining_days // 30
    remaining_days = remaining_days % 30
    days = remaining_days
    hours = difference.seconds // 3600

    sentence = f'\nThe previous scan was taken {years} years, {months} months, {days} days, {hours} hours back.'
    return sentence


class UnifiedCXRDataset(Dataset):

    def __init__(self,
                 metadata_file,
                 label_file,
                 img_root_dir,
                 text_root_dir,
                 vqgan_model_path,
                 vqgan_config_path,
                 codebook_indices_path,
                 vqgan,
                 max_img_num,
                 max_text_len,
                 tokenizer,
                 target_count,
                 target_view,
                 under_sample="fixed"
                 ):
        super().__init__()
        assert max_img_num <= target_count, f'max_img_num({max_img_num}) should be less than target_count({target_count}).'

        self.under_sample = under_sample.split('_')[0]  # fixed
        self.select_studies = under_sample.split('_')[1]  # 'each' or 'all', 'all': using all groups (S w/1, w/2, w/3), 'each': using only selected single group
        self.training_mode = under_sample.split('_')[-1]  # unified
        
        self.dict_by_subject_id = defaultdict(list)

        f = open(metadata_file, 'r')
        columns = ['idx','dicom_id','subject_id','study_id','Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices','total_number_of_studies','study_id_count']
        rdr = csv.reader(f)
        self.rd2 = pd.read_csv(label_file, names = columns) #santosh
        # print(self.rd2)
        # self.weights = np.array([0.5023, 0.4156, 0.9114, 0.6777, 0.9858, 0.8778, 0.4067, 0.2331, 0.9221, 1.0000, 0.8601])
        for i, line in tqdm(enumerate(rdr)):
            idx, dicom_id, subject_id, study_id, ViewPosition, StudyDate, StudyTime, count, curr_state = line  # [427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5,10046166,50051329,LATERAL,2]
            if self.select_studies == 'each':
                if (int(count) == int(target_count) and ViewPosition in target_view):
                    self.dict_by_subject_id[subject_id].append(line)
            elif self.select_studies == 'all':
                if (int(count) <= int(target_count) and ViewPosition in target_view):
                    self.dict_by_subject_id[subject_id].append(line)

        # print('self.dict_by_studyid', self.dict_by_studyid)
        if self.select_studies == 'all':
            self.dict_by_subject_id = {k: self.dict_by_subject_id[k] for k in self.dict_by_subject_id.keys() if len(self.dict_by_subject_id[k]) == int(self.dict_by_subject_id[k][0][-2])}
        elif self.select_studies == 'each':
            self.dict_by_subject_id = {k: self.dict_by_subject_id[k] for k in self.dict_by_subject_id.keys() if len(self.dict_by_subject_id[k]) == target_count}


        # print('self.dict_by_subject_id ', self.dict_by_subject_id)

        self.key_list = list(self.dict_by_subject_id.keys())

        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir

        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

        if vqgan == 512:
            self.img_fmap_size = 32
            self.img_reso = 512  # eg. 256 or 512 in my case
            self.img_len = 1024 + 2  # eg. 32**2 = 1024
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024

        else:
            NotImplemented

        with open(codebook_indices_path, 'rb') as f:
            self.indices_dict = pickle.load(f)

        # 2 of 3: max_img_num = 2, target_count = 3
        self.max_img_num = max_img_num
        self.target_count = target_count

        self.max_text_len = max_text_len

        self.tokenizer = tokenizer
        # self.text_vocab_size = self.tokenizer.get_vocab_size()
        self.text_vocab_size = len(self.tokenizer)
        print(self.text_vocab_size)
        # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
        self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
        ])

        self.slots = []


        self.modes = ['txt']
        for i in range(self.max_img_num):
            y = [self.img_vocab_size + i] * (self.img_len)
            self.slots.extend(y)
            self.modes.append(f'img{i + 1}')
        
        self.weights = self.get_weights()

    def get_weights(self):
        df = self.rd2[self.rd2['study_id_count']==1]
        df = df.drop(columns = ['idx', 'dicom_id', 'subject_id', 'study_id', 'total_number_of_studies','study_id_count'])
        weights = np.nansum(df, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        # print("task weights", weights)
        return weights


    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        subject_id = self.key_list[idx]
 
        if self.select_studies == 'each':
            assert len(self.dict_by_subject_id[subject_id]) == self.target_count, f'{subject_id} has {len(self.dict_by_subject_id[subject_id])} data, but target_count is {self.target_count}.'
        elif self.select_studies == 'all':
            assert len(self.dict_by_subject_id[subject_id]) <= self.target_count, f'{subject_id} has {len(self.dict_by_subject_id[subject_id])} data, but target_count is {self.target_count}.'

        if self.max_img_num == self.target_count:
            imgs_meta = self.dict_by_subject_id[subject_id]

        elif self.max_img_num < self.target_count:
            if self.under_sample == 'fixed':
                imgs_meta = self.dict_by_subject_id[subject_id][self.max_img_num:]
            elif self.under_sample == 'random':
                imgs_meta = random.sample(self.dict_by_subject_id[subject_id], self.max_img_num)

        if self.select_studies == 'all':
            num_img_in_subject = int(self.dict_by_subject_id[subject_id][0][-2])
        elif self.select_studies == 'each':
            num_img_in_subject = self.max_img_num



        # imgs
        img_paths = ''
        image_output = []
        view_position = []
        img_state = []

        for i in range(num_img_in_subject):
            # print('helloooooooooooooo',self.rd2)
            idx, dicom_id, subject_id, studyid, ViewPosition, StudyDate, StudyTime, count, curr_state = imgs_meta[i]#[:4]
            # print('jjjjjjjjjjjjjjj', dicom_id,subject_id,studyid, curr_state)
            img_path = os.path.join(self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id.split('_')[0], 's' + studyid, dicom_id + '.jpg')
            image_indices = self.indices_dict[dicom_id].copy()  # indices list
            if curr_state == '1' and ViewPosition == 'AP':
                rs = self.rd2.loc[(self.rd2['dicom_id'] == str(dicom_id)) & (self.rd2['subject_id'] == str(subject_id)) & (self.rd2['study_id']==int(studyid))]
                rs1 = rs.values.tolist()[0]
                nr1, nr2, nr3, nr4, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity, No_Finding, Pleural_Effusion, Pleural_Other, Pneumonia, Pneumothorax, Support_Devices, nr5, nr6 = [rs1[i] for i in range(len(rs.axes[1]))] #santosh
                if No_Finding == 1: #sansan
                    Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity, Pleural_Effusion, Pleural_Other, Pneumonia, Pneumothorax, Support_Devices =  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                labels = [Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged_Cardiomediastinum, Fracture, Lung_Lesion, Lung_Opacity, Pleural_Effusion, Pleural_Other, Pneumonia, Pneumothorax, Support_Devices] #sansan
                
                labels = np.array(labels)
                labels= labels.astype(float)
                labels[labels == -1] = np.nan

                image_indices.insert(0, 1025)  # self.tokenizer.token_to_id("[SOS1]")
                image_indices.append(1026)  # self.tokenizer.token_to_id("[EOS1]"
                image_output.append(torch.tensor(image_indices))
                ViewPosition = 'AP_and_curr'

            elif curr_state == '0' and ViewPosition =='AP':
                image_indices.insert(0, 1027)  # self.tokenizer.token_to_id("[SOS2]")
                image_indices.append(1028)  # self.tokenizer.token_to_id("[EOS2]")
                image_output.append(torch.tensor(image_indices))
                ViewPosition = 'AP_and_prev'

            # elif ViewPosition == 'LATERAL':
            #     image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
            #     image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
            #     image_output.append(torch.tensor(image_indices))
            # elif ViewPosition == 'LL':
            #     image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
            #     image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
            #     image_output.append(torch.tensor(image_indices))
            else:
                raise ValueError
            img_paths += (img_path + '|')
            view_position.append(ViewPosition)
            img_state.append(curr_state)

        # PAD imgs
        if num_img_in_subject < self.max_img_num:
            assert self.select_studies == 'all'
            for i in range(self.max_img_num - num_img_in_subject):
                image_indices = [1024] * self.img_len
                image_output.append(torch.tensor(image_indices))
                img_paths += ('PAD' + '|')
                view_position.append('PAD')
                img_state.append(curr_state)

            self.modes = ['txt']
            for i in range(num_img_in_subject):
                self.modes.append(f'img{i + 1}')
            random.shuffle(self.modes)
            for i in range(num_img_in_subject, self.max_img_num):
                self.modes.append(f'img{i + 1}')
        else:
            random.shuffle(self.modes)

        # report

        for i in range(num_img_in_subject):
            idx, dicom_id, subject_id, studyid, ViewPosition_text, StudyDate, StudyTime, count, curr_state = imgs_meta[i]#[:4]
            if num_img_in_subject == 1:
                count = '1'
            if curr_state == '0':#  and num_img_in_subject == 2:
                prev_StudyDate = StudyDate
                prev_StudyTime = StudyTime
                prevstudy_timestamp = datatime_conversion(prev_StudyDate, prev_StudyTime)
                # print('prevvvvvvvvvvvvvvv',prevstudy_timestamp)
                continue
            else:

                # print('in elseeeee', curr_state, num_img_in_subject)
                text_path = os.path.join(self.text_root_dir, studyid + '.txt')
                with open(text_path, 'r') as f:
                    data = f.read()
                if int(count) == 2:
                    curr_StudyDate = StudyDate
                    curr_StudyTime = StudyTime
                    currstudy_timestamp = datatime_conversion(curr_StudyDate, curr_StudyTime)
                    # print('currrrrrr',currstudy_timestamp)

                    timediff = difference(prevstudy_timestamp, currstudy_timestamp)
                    # print('sentenceeeeeeeeeeeeeeeeeee',prevstudy_timestamp, currstudy_timestamp, timediff)
                    data += timediff
                    
                    # ADAM TEMPORAL
                    dt = rem_time(currstudy_timestamp) - rem_time(prevstudy_timestamp)
                    dt_in_days = max(1, abs(float(dt.days))) # CALCULATE SOMEHOW
                    # print(dt_in_days, abs(float(dt.days)), currstudy_timestamp, prevstudy_timestamp)
                    
                else:
                    data+= "\nThere is no previous scan available."
                    
                    # ADAM TEMPORAL
                    dt_in_days = -1
                
                
                # print(data)
                #src = data.replace('  ', ' ').replace('  ', ' ').lower()
                src = self.clean_report_mimic_cxr(data)
                # ids_list = self.tokenizer.encode(src).ids
                text_output = self.tokenizer.encode(src, add_special_tokens=True, padding='max_length', max_length = 256, truncation = True, return_tensors='pt') #sansan
                # # print(text_output.shape)
                text_output = text_output.squeeze(0)
                # # print('afterrrrrrr', text_output.shape)
                #  = torch.tensor(ids_list)
                # text_output = torch.tensor(ids_list)
        # weights = UnifiedCXRDataset.get_weights(self)
        # print(weights, type(weights))
        # print(labels, type(labels))
    
        outputs = {'txt': text_output, 'modes': self.modes, 'subject_id': subject_id, 'dicom_id':dicom_id,
                   'img_paths': img_paths, 'view_position': view_position, 'image_state': img_state, 'labels':labels, 'weights':self.weights,
                   'dt_in_days': dt_in_days # ADAM TEMPORAL
                   }

        for i in range(self.max_img_num):
            outputs[f'img{i+1}'] = image_output[i]
        return outputs
    #     self.dict_by_studyid = defaultdict(list)

    #     f = open(metadata_file, 'r')
    #     rdr = csv.reader(f)
    #     for i, line in tqdm(enumerate(rdr)):
    #         dicom_id, subject_id, study_id, ViewPosition, count = line  # [427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5,10046166,50051329,LATERAL,2]
    #         if self.select_studies == 'each':
    #             if (int(count) == int(target_count) and ViewPosition in target_view):
    #                 self.dict_by_studyid[study_id].append(line)
    #         elif self.select_studies == 'all':
    #             if (int(count) <= int(target_count) and ViewPosition in target_view):
    #                 self.dict_by_studyid[study_id].append(line)

    #     if self.select_studies == 'all':
    #         self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == int(self.dict_by_studyid[k][0][-1])}
    #     elif self.select_studies == 'each':
    #         self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == target_count}

    #     self.key_list = list(self.dict_by_studyid.keys())

    #     self.img_root_dir = img_root_dir
    #     self.text_root_dir = text_root_dir

    #     self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

    #     if vqgan == 512:
    #         self.img_fmap_size = 32
    #         self.img_reso = 512  # eg. 256 or 512 in my case
    #         self.img_len = 1024 + 2  # eg. 32**2 = 1024
    #         self.img_vocab_size = self.vae.num_tokens  # eg. 1024

    #     else:
    #         NotImplemented

    #     with open(codebook_indices_path, 'rb') as f:
    #         self.indices_dict = pickle.load(f)

    #     # 2 of 3: max_img_num = 2, target_count = 3
    #     self.max_img_num = max_img_num
    #     self.target_count = target_count

    #     self.max_text_len = max_text_len

    #     self.tokenizer = tokenizer
    #     self.text_vocab_size = self.tokenizer.get_vocab_size()

    #     # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
    #     self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
    #     self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
    #     self.totensor = albumentations.pytorch.transforms.ToTensorV2()
    #     self.preprocessor = albumentations.Compose([
    #         self.rescaler,
    #         self.cropper,
    #     ])

    #     self.slots = []


    #     self.modes = ['txt']
    #     for i in range(self.max_img_num):
    #         y = [self.img_vocab_size + i] * (self.img_len)
    #         self.slots.extend(y)
    #         self.modes.append(f'img{i + 1}')

    # def __len__(self):
    #     return len(self.key_list)

    # def __getitem__(self, idx):
    #     study_id = self.key_list[idx]

    #     if self.select_studies == 'each':
    #         assert len(self.dict_by_studyid[study_id]) == self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'
    #     elif self.select_studies == 'all':
    #         assert len(self.dict_by_studyid[study_id]) <= self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'

    #     if self.max_img_num == self.target_count:
    #         imgs_meta = self.dict_by_studyid[study_id]

    #     elif self.max_img_num < self.target_count:
    #         if self.under_sample == 'fixed':
    #             imgs_meta = self.dict_by_studyid[study_id][:self.max_img_num]
    #         elif self.under_sample == 'random':
    #             imgs_meta = random.sample(self.dict_by_studyid[study_id], self.max_img_num)

    #     if self.select_studies == 'all':
    #         num_img_in_study = int(self.dict_by_studyid[study_id][0][-1])
    #     elif self.select_studies == 'each':
    #         num_img_in_study = self.max_img_num


    #     # imgs
    #     img_paths = ''
    #     image_output = []
    #     view_position = []

    #     for i in range(num_img_in_study):
    #         dicom_id, subject_id, studyid, ViewPosition = imgs_meta[i][:4]
    #         img_path = os.path.join(self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id, 's' + studyid, dicom_id + '.jpg')
    #         image_indices = self.indices_dict[dicom_id].copy()  # indices list
    #         if ViewPosition == 'AP':
    #             image_indices.insert(0, 1025)  # self.tokenizer.token_to_id("[SOS1]")
    #             image_indices.append(1026)  # self.tokenizer.token_to_id("[EOS1]"
    #             image_output.append(torch.tensor(image_indices))
    #         elif ViewPosition == 'PA':
    #             image_indices.insert(0, 1027)  # self.tokenizer.token_to_id("[SOS2]")
    #             image_indices.append(1028)  # self.tokenizer.token_to_id("[EOS2]")
    #             image_output.append(torch.tensor(image_indices))
    #         elif ViewPosition == 'LATERAL':
    #             image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
    #             image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
    #             image_output.append(torch.tensor(image_indices))
    #         elif ViewPosition == 'LL':
    #             image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
    #             image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
    #             image_output.append(torch.tensor(image_indices))
    #         else:
    #             raise ValueError
    #         img_paths += (img_path + '|')
    #         view_position.append(ViewPosition)

    #     # PAD imgs
    #     if num_img_in_study < self.max_img_num:
    #         assert self.select_studies == 'all'
    #         for i in range(self.max_img_num - num_img_in_study):
    #             image_indices = [1024] * self.img_len
    #             image_output.append(torch.tensor(image_indices))
    #             img_paths += ('PAD' + '|')
    #             view_position.append('PAD')

    #         self.modes = ['txt']
    #         for i in range(num_img_in_study):
    #             self.modes.append(f'img{i + 1}')
    #         random.shuffle(self.modes)
    #         for i in range(num_img_in_study, self.max_img_num):
    #             self.modes.append(f'img{i + 1}')
    #     else:
    #         random.shuffle(self.modes)

    #     # report
    #     text_path = os.path.join(self.text_root_dir, 's' + study_id + '.txt')
    #     with open(text_path, 'r') as f:
    #         data = f.read()
    #     src = data.replace('  ', ' ').replace('  ', ' ').lower()
    #     ids_list = self.tokenizer.encode(src).ids
    #     text_output = torch.tensor(ids_list)


    #     outputs = {'txt': text_output, 'modes': self.modes, 'study_id': study_id,
    #                'img_paths': img_paths, 'view_position': view_position}

    #     for i in range(self.max_img_num):
    #         outputs[f'img{i+1}'] = image_output[i]
    #     return outputs



# '''
# The upper code is for training

# SEPARATION

# The below is for testing
# '''

# import os
# import csv
# import random
# import pickle
# from tqdm import tqdm
# from collections import defaultdict

# import albumentations
# import albumentations.pytorch

# import torch
# from torch.utils.data import Dataset

# from vae import VQGanVAE

# random.seed(42)

# class UnifiedCXRDataset(Dataset):

#     def __init__(self,
#                  metadata_file,
#                  img_root_dir,
#                  text_root_dir,
#                  vqgan_model_path,
#                  vqgan_config_path,
#                  codebook_indices_path,
#                  vqgan,
#                  max_img_num,
#                  max_text_len,
#                  tokenizer,
#                  target_count,
#                  target_view,
#                  under_sample="fixed"
#                  ):
#         super().__init__()

#         assert max_img_num <= target_count, f'max_img_num({max_img_num}) should be less than target_count({target_count}).'

#         self.under_sample = under_sample.split('_')[0]  # fixed
#         self.select_studies = under_sample.split('_')[1]  # 'each' or 'all', 'all': using all groups (S w/1, w/2, w/3), 'each': using only selected single group
#         self.training_mode = under_sample.split('_')[-1]  # unified
        
#         self.dict_by_subject_id = defaultdict(list)

#         f = open(metadata_file, 'r')
#         rdr = csv.reader(f)
#         for i, line in tqdm(enumerate(rdr)):
#             idx, dicom_id, subject_id, study_id, ViewPosition, StudyDate, StudyTime, count, curr_state = line  # [427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5,10046166,50051329,LATERAL,2]
#             if self.select_studies == 'each':
#                 if (int(count) == int(target_count) and ViewPosition in target_view):
#                     self.dict_by_subject_id[subject_id].append(line)
#             elif self.select_studies == 'all':
#                 if (int(count) <= int(target_count) and ViewPosition in target_view):
#                     self.dict_by_subject_id[subject_id].append(line)

#         # print('self.dict_by_studyid', self.dict_by_studyid)
#         if self.select_studies == 'all':
#             self.dict_by_subject_id = {k: self.dict_by_subject_id[k] for k in self.dict_by_subject_id.keys() if len(self.dict_by_subject_id[k]) == int(self.dict_by_subject_id[k][0][-2])}
#         elif self.select_studies == 'each':
#             self.dict_by_subject_id = {k: self.dict_by_subject_id[k] for k in self.dict_by_subject_id.keys() if len(self.dict_by_subject_id[k]) == target_count}


#         # print('self.dict_by_studyid', self.dict_by_studyid)

#         self.key_list = list(self.dict_by_subject_id.keys())

#         self.img_root_dir = img_root_dir
#         self.text_root_dir = text_root_dir

#         self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

#         if vqgan == 512:
#             self.img_fmap_size = 32
#             self.img_reso = 512  # eg. 256 or 512 in my case
#             self.img_len = 1024 + 2  # eg. 32**2 = 1024
#             self.img_vocab_size = self.vae.num_tokens  # eg. 1024

#         else:
#             NotImplemented

#         with open(codebook_indices_path, 'rb') as f:
#             self.indices_dict = pickle.load(f)

#         # 2 of 3: max_img_num = 2, target_count = 3
#         self.max_img_num = max_img_num
#         self.target_count = target_count

#         self.max_text_len = max_text_len

#         self.tokenizer = tokenizer
#         self.text_vocab_size = self.tokenizer.get_vocab_size()

#         # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
#         self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
#         self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
#         self.totensor = albumentations.pytorch.transforms.ToTensorV2()
#         self.preprocessor = albumentations.Compose([
#             self.rescaler,
#             self.cropper,
#         ])

#         self.slots = []


#         self.modes = ['txt']
#         for i in range(self.max_img_num):
#             y = [self.img_vocab_size + i] * (self.img_len)
#             self.slots.extend(y)
#             self.modes.append(f'img{i + 1}')

#     def __len__(self):
#         return len(self.key_list)

#     def __getitem__(self, idx):
#         subject_id = self.key_list[idx]

#         if self.select_studies == 'each':
#             assert len(self.dict_by_subject_id[subject_id]) == self.target_count, f'{subject_id} has {len(self.dict_by_subject_id[subject_id])} data, but target_count is {self.target_count}.'
#         elif self.select_studies == 'all':
#             assert len(self.dict_by_subject_id[subject_id]) <= self.target_count, f'{subject_id} has {len(self.dict_by_subject_id[subject_id])} data, but target_count is {self.target_count}.'

#         if self.max_img_num == self.target_count:
#             imgs_meta = self.dict_by_subject_id[subject_id]

#         elif self.max_img_num < self.target_count:
#             if self.under_sample == 'fixed':
#                 imgs_meta = self.dict_by_subject_id[subject_id][self.max_img_num:]
#             elif self.under_sample == 'random':
#                 imgs_meta = random.sample(self.dict_by_subject_id[subject_id], self.max_img_num)

#         if self.select_studies == 'all':
#             num_img_in_subject = int(self.dict_by_subject_id[subject_id][0][-2])
#         elif self.select_studies == 'each':
#             num_img_in_subject = self.max_img_num



#         # imgs
#         img_paths = ''
#         image_output = []
#         view_position = []
#         img_state = []

#         for i in range(num_img_in_subject):
#             idx, dicom_id, subject_id, studyid, ViewPosition, StudyDate, StudyTime, count, curr_state = imgs_meta[i]#[:4]
#             img_path = os.path.join(self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id.split('_')[0], 's' + studyid, dicom_id + '.jpg')
#             image_indices = self.indices_dict[dicom_id].copy()  # indices list
#             if curr_state == '1' and ViewPosition == 'AP':
#                 image_indices.insert(0, 1025)  # self.tokenizer.token_to_id("[SOS1]")
#                 image_indices.append(1026)  # self.tokenizer.token_to_id("[EOS1]"
#                 image_output.append(torch.tensor(image_indices))
#                 ViewPosition = 'AP_and_curr'

#             elif curr_state == '0' and ViewPosition =='AP':
#                 image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS2]")
#                 image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS2]")
#                 image_output.append(torch.tensor(image_indices))
#                 ViewPosition = 'AP_and_prev'

#             # elif ViewPosition == 'LATERAL':
#             #     image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
#             #     image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
#             #     image_output.append(torch.tensor(image_indices))
#             # elif ViewPosition == 'LL':
#             #     image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
#             #     image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
#             #     image_output.append(torch.tensor(image_indices))
#             else:
#                 raise ValueError
#             img_paths += (img_path + '|')
#             view_position.append(ViewPosition)
#             img_state.append(curr_state)

#         # PAD imgs
#         if num_img_in_subject < self.max_img_num:
#             assert self.select_studies == 'all'
#             for i in range(self.max_img_num - num_img_in_subject):
#                 image_indices = [1024] * self.img_len
#                 image_output.append(torch.tensor(image_indices))
#                 img_paths += ('PAD' + '|')
#                 view_position.append('PAD')
#                 img_state.append(curr_state)

#             self.modes = ['txt']
#             for i in range(num_img_in_subject):
#                 self.modes.append(f'img{i + 1}')
#             random.shuffle(self.modes)
#             for i in range(num_img_in_subject, self.max_img_num):
#                 self.modes.append(f'img{i + 1}')
#         else:
#             random.shuffle(self.modes)

#         # report

#         for i in range(num_img_in_subject):
#             idx, dicom_id, subject_id, studyid, ViewPosition_text, StudyDate, StudyTime, count, curr_state = imgs_meta[i]#[:4]
#             if curr_state == '0':#  and num_img_in_subject == 2:
#                 continue
#             else:
#                 # print('in elseeeee', curr_state, num_img_in_subject)
#                 text_path = os.path.join(self.text_root_dir, studyid + '.txt')
#                 with open(text_path, 'r') as f:
#                     data = f.read()
#                 src = data.replace('  ', ' ').replace('  ', ' ').lower()
#                 ids_list = self.tokenizer.encode(src).ids
#                 # text_output = self.tokenizer.encode(src, add_special_tokens=True, padding='max_length', max_length = 256, truncation = True, return_tensors='pt')
#                 # # print(text_output.shape)
#                 # text_output = text_output.squeeze(0)
#                 # # print('afterrrrrrr', text_output.shape)
#                 #  = torch.tensor(ids_list)
#                 text_output = torch.tensor(ids_list)


#         outputs = {'txt': text_output, 'modes': self.modes, 'subject_id': subject_id, 'dicom_id':dicom_id,
#                    'img_paths': img_paths, 'view_position': view_position, 'image_state': img_state}

#         for i in range(self.max_img_num):
#             outputs[f'img{i+1}'] = image_output[i]
#         return outputs

#     #     self.dict_by_studyid = defaultdict(list)

#     #     f = open(metadata_file, 'r')
#     #     rdr = csv.reader(f)
#     #     for i, line in tqdm(enumerate(rdr)):
#     #         dicom_id, subject_id, study_id, ViewPosition, count = line  # [427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5,10046166,50051329,LATERAL,2]
#     #         if self.select_studies == 'each':
#     #             if (int(count) == int(target_count) and ViewPosition in target_view):
#     #                 self.dict_by_studyid[study_id].append(line)
#     #         elif self.select_studies == 'all':
#     #             if (int(count) <= int(target_count) and ViewPosition in target_view):
#     #                 self.dict_by_studyid[study_id].append(line)

#     #     if self.select_studies == 'all':
#     #         self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == int(self.dict_by_studyid[k][0][-1])}
#     #     elif self.select_studies == 'each':
#     #         self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == target_count}

#     #     self.key_list = list(self.dict_by_studyid.keys())

#     #     self.img_root_dir = img_root_dir
#     #     self.text_root_dir = text_root_dir

#     #     self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

#     #     if vqgan == 512:
#     #         self.img_fmap_size = 32
#     #         self.img_reso = 512  # eg. 256 or 512 in my case
#     #         self.img_len = 1024 + 2  # eg. 32**2 = 1024
#     #         self.img_vocab_size = self.vae.num_tokens  # eg. 1024

#     #     else:
#     #         NotImplemented

#     #     with open(codebook_indices_path, 'rb') as f:
#     #         self.indices_dict = pickle.load(f)

#     #     # 2 of 3: max_img_num = 2, target_count = 3
#     #     self.max_img_num = max_img_num
#     #     self.target_count = target_count

#     #     self.max_text_len = max_text_len

#     #     self.tokenizer = tokenizer
#     #     self.text_vocab_size = self.tokenizer.get_vocab_size()

#     #     # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
#     #     self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
#     #     self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
#     #     self.totensor = albumentations.pytorch.transforms.ToTensorV2()
#     #     self.preprocessor = albumentations.Compose([
#     #         self.rescaler,
#     #         self.cropper,
#     #     ])

#     #     self.slots = []


#     #     self.modes = ['txt']
#     #     for i in range(self.max_img_num):
#     #         y = [self.img_vocab_size + i] * (self.img_len)
#     #         self.slots.extend(y)
#     #         self.modes.append(f'img{i + 1}')

#     # def __len__(self):
#     #     return len(self.key_list)

#     # def __getitem__(self, idx):
#     #     study_id = self.key_list[idx]

#     #     if self.select_studies == 'each':
#     #         assert len(self.dict_by_studyid[study_id]) == self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'
#     #     elif self.select_studies == 'all':
#     #         assert len(self.dict_by_studyid[study_id]) <= self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'

#     #     if self.max_img_num == self.target_count:
#     #         imgs_meta = self.dict_by_studyid[study_id]

#     #     elif self.max_img_num < self.target_count:
#     #         if self.under_sample == 'fixed':
#     #             imgs_meta = self.dict_by_studyid[study_id][:self.max_img_num]
#     #         elif self.under_sample == 'random':
#     #             imgs_meta = random.sample(self.dict_by_studyid[study_id], self.max_img_num)

#     #     if self.select_studies == 'all':
#     #         num_img_in_study = int(self.dict_by_studyid[study_id][0][-1])
#     #     elif self.select_studies == 'each':
#     #         num_img_in_study = self.max_img_num


#     #     # imgs
#     #     img_paths = ''
#     #     image_output = []
#     #     view_position = []

#     #     for i in range(num_img_in_study):
#     #         dicom_id, subject_id, studyid, ViewPosition = imgs_meta[i][:4]
#     #         img_path = os.path.join(self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id, 's' + studyid, dicom_id + '.jpg')
#     #         image_indices = self.indices_dict[dicom_id].copy()  # indices list
#     #         if ViewPosition == 'AP':
#     #             image_indices.insert(0, 1025)  # self.tokenizer.token_to_id("[SOS1]")
#     #             image_indices.append(1026)  # self.tokenizer.token_to_id("[EOS1]"
#     #             image_output.append(torch.tensor(image_indices))
#     #         elif ViewPosition == 'PA':
#     #             image_indices.insert(0, 1027)  # self.tokenizer.token_to_id("[SOS2]")
#     #             image_indices.append(1028)  # self.tokenizer.token_to_id("[EOS2]")
#     #             image_output.append(torch.tensor(image_indices))
#     #         elif ViewPosition == 'LATERAL':
#     #             image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
#     #             image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
#     #             image_output.append(torch.tensor(image_indices))
#     #         elif ViewPosition == 'LL':
#     #             image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
#     #             image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
#     #             image_output.append(torch.tensor(image_indices))
#     #         else:
#     #             raise ValueError
#     #         img_paths += (img_path + '|')
#     #         view_position.append(ViewPosition)

#     #     # PAD imgs
#     #     if num_img_in_study < self.max_img_num:
#     #         assert self.select_studies == 'all'
#     #         for i in range(self.max_img_num - num_img_in_study):
#     #             image_indices = [1024] * self.img_len
#     #             image_output.append(torch.tensor(image_indices))
#     #             img_paths += ('PAD' + '|')
#     #             view_position.append('PAD')

#     #         self.modes = ['txt']
#     #         for i in range(num_img_in_study):
#     #             self.modes.append(f'img{i + 1}')
#     #         random.shuffle(self.modes)
#     #         for i in range(num_img_in_study, self.max_img_num):
#     #             self.modes.append(f'img{i + 1}')
#     #     else:
#     #         random.shuffle(self.modes)

#     #     # report
#     #     text_path = os.path.join(self.text_root_dir, 's' + study_id + '.txt')
#     #     with open(text_path, 'r') as f:
#     #         data = f.read()
#     #     src = data.replace('  ', ' ').replace('  ', ' ').lower()
#     #     ids_list = self.tokenizer.encode(src).ids
#     #     text_output = torch.tensor(ids_list)


#     #     outputs = {'txt': text_output, 'modes': self.modes, 'study_id': study_id,
#     #                'img_paths': img_paths, 'view_position': view_position}

#     #     for i in range(self.max_img_num):
#     #         outputs[f'img{i+1}'] = image_output[i]
#     #     return outputs