import pandas as pd
import os
from tqdm import tqdm
import logging
import math
import numpy as np
import pickle

from CONFIG import DITEM_ID_IN_SCOPE,DLABITEM_ID_IN_SCOPE,INPUT_TOTAL_EQUIVALENT_VOL_COEF,\
    PRE_ADM_FLUID_INTAKE,VASOPRESSORS_RATE_CONVERSION,VASOPRESSORS_ITEMIDS,OUTPUT_ITEMID

class DataProcessor():
    def __init__(self,folder_path,to_save_folder_path = '',
                 los_lower_bound = 5,los_upper_bound = 20,
                 age_lower_bound = 18, age_upper_bound = 89) -> None:
        self.FOLDER_PATH = folder_path
        self.ICU_LOS_LOWER_BOUND = los_lower_bound
        self.ICU_LOS_UPPER_BOUND = los_upper_bound
        self.AGE_LOWER_BOUND = age_lower_bound
        self.AGE_UPPER_BOUND = age_upper_bound
        self.SAVE_FLAG = False
        self.TO_SAVE_FOLDER_PATH = to_save_folder_path
        self.check_path(folder_path,to_save_folder_path)

    def check_path(self,folder_path,to_save_folder_path):
        if not os.path.exists(folder_path):
            logging.warning('Folder does not exist: ', folder_path)
        if to_save_folder_path != '':
            self.SAVE_FLAG = True
            if not os.path.exists(to_save_folder_path):
                os.makedirs(to_save_folder_path)
            logging.info('Processed data will be stored at: ',to_save_folder_path)

    def process(self,model_name = '',sampling = 1):
        self.load_items_mapping()
        self.load_patients()
        self.load_admissions()
        self.load_icustays()
        self.filter_icustays_admissions_patients()
        self.prepare_meta_table(sampling)
        self.load_inputevents()
        self.load_outputevents()
        self.load_chartevents()
        self.load_labevents()
        #self.pivot_table()
        if self.SAVE_FLAG:
            if model_name == 'strats':
                self.prepare_strats_input(0.7,0.2,0.1)
            else:
                self.save_results_csv()

    def load_items_mapping(self):
        self.ditems = pd.read_csv(os.path.join(self.FOLDER_PATH,'D_ITEMS.csv.gz'), 
                                usecols = ['ITEMID','LABEL'],
                                compression='gzip')
        self.dlabitems = pd.read_csv(os.path.join(self.FOLDER_PATH,'D_LABITEMS.csv.gz'), 
                                usecols = ['ITEMID','LABEL'],
                                compression='gzip')
        self.items_info = pd.concat([self.ditems,self.dlabitems])

    def load_patients(self):
        self.patients = pd.read_csv(os.path.join(self.FOLDER_PATH, 'PATIENTS.csv.gz'),
                                    usecols=['SUBJECT_ID', 'DOB', 'DOD', 'GENDER'],
                                    compression='gzip')
        #sanity check
        if len(self.patients) != self.patients['SUBJECT_ID'].nunique():
            print('Duplicate patient ID!')
        self.patients['DOB'] = pd.to_datetime(self.patients['DOB'])
        self.patients['DOD'] = pd.to_datetime(self.patients['DOD'])
    
    def load_admissions(self):
        self.admissions = pd.read_csv(os.path.join(self.FOLDER_PATH, 'ADMISSIONS.csv.gz'),
                                      usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DEATHTIME','HOSPITAL_EXPIRE_FLAG'],
                                      compression='gzip')
        self.admissions = self.admissions.rename(columns={'HOSPITAL_EXPIRE_FLAG': 'IN_HOS_MORTALITY'})

    def load_icustays(self):
        self.icustays = pd.read_csv(os.path.join(self.FOLDER_PATH, 'ICUSTAYS.csv.gz'),
                                    usecols=['SUBJECT_ID','HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME','LOS'],
                                    compression='gzip')
        
    def filter_icustays_admissions_patients(self):
        self.icustays = self.icustays.merge(self.patients, on = 'SUBJECT_ID', how = 'left')
        self.icustays['INTIME'] = pd.to_datetime(self.icustays['INTIME'])
        #filter out too short stays
        self.icustays = self.icustays[(self.icustays['LOS'] >= self.ICU_LOS_LOWER_BOUND) & (self.icustays['LOS'] <=self.ICU_LOS_UPPER_BOUND)]
        #calculate age and filter out non-adults/elders
        self.icustays['AGE'] = self.icustays['INTIME'].map(lambda x:x.year) - self.icustays['DOB'].map(lambda x:x.year)
        self.icustays = self.icustays[(self.icustays['AGE'] >= self.AGE_LOWER_BOUND) & (self.icustays['AGE'] <=self.AGE_UPPER_BOUND)]

    def prepare_meta_table(self,sampling):
        #join icustays and admissions(only look at icustays have admission data)
        self.meta_table = pd.merge(self.icustays,self.admissions, on=['SUBJECT_ID','HADM_ID'])
        #select the first icu stay for the same patient if there are multiple so that patients are unique
        self.meta_table = self.meta_table.groupby(by = ['SUBJECT_ID']).agg({'HADM_ID':'first',
                                                                            'ICUSTAY_ID':'first',
                                                                            'ADMITTIME':'first',
                                                                            'DISCHTIME':'first',
                                                                            'DEATHTIME':'first',
                                                                            'IN_HOS_MORTALITY':'first',
                                                                            'INTIME':'first',
                                                                            'LOS':'first',
                                                                            'GENDER':'first',
                                                                            'AGE':'first',
                                                                            }).reset_index()
        self.meta_table = pd.merge(self.meta_table,self.patients, on='SUBJECT_ID')
        if sampling != 1:
            total_size = len(self.meta_table)
            sample_size = int(total_size*sampling)
            samples_ids = self.meta_table['ICUSTAY_ID']
            np.random.seed(0)
            np.random.shuffle(samples_ids)
            selected_ids = samples_ids[:sample_size]
            self.meta_table = self.meta_table.loc[self.meta_table['ICUSTAY_ID'].isin(selected_ids)]
        

    def load_chartevents(self):
        logging.info('******** Loading chart events ********')
        chartevents = []
        for chunk in tqdm(pd.read_csv(os.path.join(self.FOLDER_PATH,'CHARTEVENTS.csv.gz'), 
                            chunksize=10000000,
                            usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 
                           'VALUE', 'VALUENUM', 'VALUEUOM', 'ERROR'],
                           compression='gzip')):
            chunk = chunk.loc[chunk['ICUSTAY_ID'].isin(self.meta_table['ICUSTAY_ID'])]
            chunk = chunk.loc[chunk['ERROR']!=1]
            chunk = chunk.loc[chunk['CHARTTIME'].notna()]
            chunk = chunk.loc[chunk['VALUENUM'].notna()]
            chunk.drop(columns=['ERROR'], inplace=True)
            chunk.loc[:,'IN_SCOPE'] = False
            #extract items in scope
            for key in DITEM_ID_IN_SCOPE:
                row_indexer = chunk['ITEMID'].isin(DITEM_ID_IN_SCOPE[key]['ids'])
                chunk.loc[row_indexer,'LABEL'] = DITEM_ID_IN_SCOPE[key]['label']
                chunk.loc[row_indexer,'IN_SCOPE'] = True
            chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'])
            chartevents.append(chunk)
        chartevents = pd.concat(chartevents)
        self.chartevents = chartevents.loc[chartevents['IN_SCOPE'] == True]
        self.chartevents.drop(columns = ['IN_SCOPE'],inplace=True)
        logging.info('******** Loading finished ********')
        
    def load_labevents(self):
        logging.info('******** Loading lab events ********')
        labevents = pd.read_csv(os.path.join(self.FOLDER_PATH, 'LABEVENTS.csv.gz'),compression='gzip')
        labevents = labevents.loc[labevents['HADM_ID'].isin(self.meta_table['HADM_ID'])]
        labevents = labevents.loc[labevents['VALUENUM'].notna()]
        labevents['CHARTTIME'] = pd.to_datetime(labevents['CHARTTIME'])
        labevents.loc[:,'IN_SCOPE'] = False
        for key in DLABITEM_ID_IN_SCOPE:
            row_indexer = labevents['ITEMID'].isin(DLABITEM_ID_IN_SCOPE[key]['ids'])
            labevents.loc[row_indexer,'LABEL'] = DLABITEM_ID_IN_SCOPE[key]['label']
            labevents.loc[row_indexer,'IN_SCOPE'] = True
        self.labevents = labevents.loc[labevents['IN_SCOPE'] == True]
        self.labevents.drop(columns = ['IN_SCOPE'],inplace=True)
        logging.info('******** Loading finished ********')

    def load_inputevents(self):
        logging.info('******** Loading input events ********')
        input_cv = self.load_inputevents_cv()
        input_mv = self.load_inputevents_mv()
        input = pd.concat([input_cv, input_mv]).reset_index(drop=True)
        self.input_real_time_standard = self.standardize_real_time_input(input)
        self.input_vasopressor_standard = self.standardize_vasopressor(input)
        self.vasopressor_convert_rate_to_amount()
        self.input_pre_admission = self.gen_pre_admission_intake(input_cv,input_mv)
        logging.info('******** Loading finished ********')

    def load_inputevents_cv(self):
        #load input from carevue
        input_cv = pd.read_csv(os.path.join(self.FOLDER_PATH,'INPUTEVENTS_CV.csv.gz'), 
                               usecols = ['ICUSTAY_ID','ITEMID','CHARTTIME','ITEMID','AMOUNT','AMOUNTUOM','RATE','RATEUOM'],
                                compression='gzip')
        input_cv = input_cv.loc[input_cv['ICUSTAY_ID'].isin(self.meta_table['ICUSTAY_ID'])]
        input_cv = input_cv.loc[input_cv['CHARTTIME'].notna()]
        input_cv['CHARTTIME'] = pd.to_datetime(input_cv['CHARTTIME'])
        return input_cv

    def load_inputevents_mv(self):
        #load input from metavision
        input_mv = pd.read_csv(os.path.join(self.FOLDER_PATH,'INPUTEVENTS_MV.csv.gz'), 
                               usecols = ['ICUSTAY_ID','ITEMID','STARTTIME','ENDTIME','ITEMID','AMOUNT','AMOUNTUOM','RATE','RATEUOM'],
                                compression='gzip')
        input_mv = input_mv.loc[input_mv['ICUSTAY_ID'].isin(self.meta_table['ICUSTAY_ID'])]
        nput_mv = input_mv.loc[input_mv['STARTTIME'].notna()]
        input_mv = input_mv.loc[input_mv['ENDTIME'].notna()]
        input_mv['STARTTIME'] = pd.to_datetime(input_mv['STARTTIME'])
        input_mv['ENDTIME'] = pd.to_datetime(input_mv['ENDTIME'])
        return input_mv

    def load_outputevents(self):
        output = pd.read_csv(os.path.join(self.FOLDER_PATH,'OUTPUTEVENTS.csv.gz'), 
                               usecols = ['ICUSTAY_ID','ITEMID','CHARTTIME','ITEMID','VALUE','VALUEUOM'],
                                compression='gzip')
        output['CHARTTIME'] = pd.to_datetime(output['CHARTTIME'])
        output = output.loc[output['ICUSTAY_ID'].notna()]
        output = output.loc[output['VALUE'].notna()]
        self.output_pre_admission = output.loc[output['ITEMID'].isin(OUTPUT_ITEMID['pre_admission']['ids'])]
        self.output_pre_admission['LABEL'] = OUTPUT_ITEMID['pre_admission']['label']
        self.output_real_time = output.loc[output['ITEMID'].isin(OUTPUT_ITEMID['real_time']['ids'])]
        self.output_real_time['LABEL'] = OUTPUT_ITEMID['real_time']['label']

    def standardize_real_time_input(self, input):
        input_real_time = input.loc[input['ITEMID'].isin(INPUT_TOTAL_EQUIVALENT_VOL_COEF)]
        input_real_time = input_real_time.loc[input_real_time['AMOUNT'].notna()]
        for key in INPUT_TOTAL_EQUIVALENT_VOL_COEF:
            row_indexer = input_real_time['ITEMID'] == key
            #total equivalent value
            input_real_time.loc[row_indexer,'TEV'] = INPUT_TOTAL_EQUIVALENT_VOL_COEF[key]*input_real_time.loc[row_indexer,'AMOUNT']
        input_real_time['CHARTTIME'] = input_real_time['CHARTTIME'].fillna(input_real_time['ENDTIME'])
        return input_real_time
    
    def gen_pre_admission_intake(self, input_cv,input_mv):
        cv_filtered = input_cv.loc[input_cv['ITEMID'].isin(PRE_ADM_FLUID_INTAKE)]
        cv_filtered = cv_filtered.groupby('ICUSTAY_ID')['AMOUNT'].sum().reset_index()
        mv_filtered = input_mv.loc[input_mv['ITEMID'].isin(PRE_ADM_FLUID_INTAKE)]
        mv_filtered = mv_filtered.groupby('ICUSTAY_ID')['AMOUNT'].sum().reset_index()
        all_filtered = pd.concat([cv_filtered,mv_filtered])
        return all_filtered

    def standardize_vasopressor(self, input):
        vaso_input = input.loc[input['ITEMID'].isin(VASOPRESSORS_ITEMIDS)&(input['RATE'].notnull())]
        vaso_input['RATE_STD'] = vaso_input.apply(self.standardize_rate, axis=1)
        return vaso_input

    def standardize_rate(self, row):
        itemid = row['ITEMID']
        rateuom = row['RATEUOM']
        rate = row['RATE']
        if (itemid, rateuom) in VASOPRESSORS_RATE_CONVERSION:
            return round(rate * VASOPRESSORS_RATE_CONVERSION[(itemid, rateuom)], 3)
        elif itemid in [30051,222315] and rate > 0.2:
            return round(rate * VASOPRESSORS_RATE_CONVERSION[(itemid, '> 0.2')], 3)
        else:
            return None
        
    def vasopressor_convert_rate_to_amount(self):
        input_vasopressor_mv = self.input_vasopressor_standard.loc[self.input_vasopressor_standard['STARTTIME'].notna()]
        input_vasopressor_mv['TD'] = input_vasopressor_mv['ENDTIME']- input_vasopressor_mv['STARTTIME']
        input_vasopressor_mv['AMOUNT'] = 0
        input_vasopressor_mv['TIME(HR)'] = 0
        new_df = []
        for _,row in tqdm(input_vasopressor_mv.iterrows()):
            id, itemid, starttime,endtime, rate, td = row['ICUSTAY_ID'],row['ITEMID'],row['STARTTIME'],row['ENDTIME'],row['RATE_STD'],row['TD']
            timediff_hour = math.floor(td.total_seconds()/3600)
            amount_hourly = (td.total_seconds()/60)*rate
            for i in range(1,int(timediff_hour)+1):
                new_df.append([id, itemid, starttime, endtime, rate, i, amount_hourly])
            res_mins = (td.total_seconds()/60) % 60
            if res_mins>0:
                amount_res = res_mins * rate
                new_df.append([id, itemid, starttime, endtime, rate, int(timediff_hour)+1, amount_res])
        input_vasopressor_standard_mv_new = pd.DataFrame(new_df, columns=['ICUSTAY_ID','ITEMID','STARTTIME','ENDTIME','RATE_STD','TIME(HR)','AMOUNT'])
        input_vasopressor_standard_new = pd.concat([self.input_vasopressor_standard.loc[self.input_vasopressor_standard['STARTTIME'].isna()],input_vasopressor_standard_mv_new])
        self.input_vasopressor_standard = input_vasopressor_standard_new
        
    def pivot_table(self):
        self.chartevents = self.chartevents.pivot_table(index=['ICUSTAY_ID','TIME(HR)'], columns='LABEL', values='VALUENUM').reset_index()
        self.labevents = self.labevents.pivot_table(index=['HADM_ID','TIME(HR)'], columns='LABEL', values='VALUENUM').reset_index()
        self.input_real_time_standard = self.input_real_time_standard.pivot_table(index=['ICUSTAY_ID','TIME(HR)'], columns='ITEMID', values='TEV').reset_index()
        #self.input_vasopressor_standard = self.input_vasopressor_standard.pivot_table(index=['ICUSTAY_ID','TIME(HR)'], columns='ITEMID', values='AMOUNT').reset_index()
        self.output_real_time = self.output_real_time.pivot_table(index=['ICUSTAY_ID','TIME(HR)'], columns='LABEL', values='VALUE').reset_index()
        self.output_pre_admission = self.output_pre_admission.pivot_table(index=['ICUSTAY_ID','TIME(HR)'], columns='LABEL', values='VALUE').reset_index()       

    def convert_events_time_to_minute(self):
        self.events.sort_values(by =['ts_id','CHARTTIME'],inplace=True)
        self.events['minute'] = (self.events['CHARTTIME'] - self.events.groupby('ts_id')['CHARTTIME'].transform('first')).dt.total_seconds() / 60
        self.events['minute'] = self.events['minute'].apply(math.ceil)
        self.events.drop(columns=['CHARTTIME'],inplace=True)

    def map_itemid_to_label(self,row):
        variable = row['variable']
        if isinstance(variable, int):
            variable_name = self.items_info.loc[self.items_info['ITEMID'] == variable, 'LABEL'].values[0]
            return variable_name
        else:
            return variable
        

    def prepare_strats_input(self, train_frac, valid_frac, test_frac):
        self.chartevents = self.chartevents[['ICUSTAY_ID','CHARTTIME','LABEL','VALUENUM']]
        self.chartevents['TABLE'] = 'chart'
        self.chartevents = self.chartevents.rename(columns={'ICUSTAY_ID':'ts_id','LABEL':'variable','VALUENUM':'value'})
        self.labevents = pd.merge(self.labevents,self.icustays, on=['SUBJECT_ID','HADM_ID'])
        self.labevents = self.labevents[['ICUSTAY_ID','CHARTTIME','LABEL','VALUENUM']]
        self.labevents = self.labevents.rename(columns={'ICUSTAY_ID':'ts_id','LABEL':'variable','VALUENUM':'value'})
        self.labevents['TABLE'] = 'lab'
        self.input_real_time_standard = self.input_real_time_standard[['ICUSTAY_ID','CHARTTIME','ITEMID','TEV']]
        self.input_real_time_standard['TABLE'] = 'input_real_time'
        self.input_real_time_standard = self.input_real_time_standard.rename(columns={'ICUSTAY_ID':'ts_id','ITEMID':'variable','TEV':'value'})
        self.output_real_time = self.output_real_time[['ICUSTAY_ID','CHARTTIME','LABEL','VALUE']]
        self.output_real_time['TABLE'] = 'output_real_time'
        self.output_real_time = self.output_real_time.rename(columns={'ICUSTAY_ID':'ts_id','LABEL':'variable','VALUE':'value'})
        self.output_pre_admission = self.output_pre_admission[['ICUSTAY_ID','CHARTTIME','LABEL','VALUE']]
        self.output_pre_admission['TABLE'] = 'output_pre_admission'
        self.output_pre_admission = self.output_pre_admission.rename(columns={'ICUSTAY_ID':'ts_id','LABEL':'variable','VALUE':'value'})
        self.events = pd.concat([self.chartevents,self.labevents,self.input_real_time_standard,self.output_real_time,self.output_pre_admission])
        self.convert_events_time_to_minute()
        self.events['variable'] = self.events.apply(self.map_itemid_to_label, axis=1)
        self.add_static_variable_to_events()

        oc = self.meta_table[['ICUSTAY_ID','HADM_ID','SUBJECT_ID','IN_HOS_MORTALITY']]
        oc = oc.rename(columns={'ICUSTAY_ID':'ts_id','IN_HOS_MORTALITY':'in_hospital_mortality'})

        all_ic_ids = self.meta_table['ICUSTAY_ID']
        np.random.seed(0)
        np.random.shuffle(all_ic_ids)
        total_num = len(all_ic_ids)
        train_ids = np.array(all_ic_ids[:int(total_num*train_frac)])
        valid_ids = np.array(all_ic_ids[int(total_num*train_frac):int(total_num*(train_frac+valid_frac))])
        test_ids = np.array(all_ic_ids[int(total_num*(train_frac+valid_frac)):])

        pickle.dump([self.events, oc, train_ids, valid_ids, test_ids], open(os.path.join(self.TO_SAVE_FOLDER_PATH,'mimic_iii.pkl'),'wb'))

    def add_static_variable_to_events(self):
        self.icustays.rename(columns={'ICUSTAY_ID':'ts_id'}, inplace=True)
        data_age = self.icustays[['ts_id', 'AGE']]
        data_age['variable'] = 'Age'
        data_age.rename(columns={'AGE':'value'}, inplace=True)
        data_gen = self.icustays[['ts_id', 'GENDER']]
        data_gen.loc[data_gen.GENDER=='M', 'GENDER'] = 0
        data_gen.loc[data_gen.GENDER=='F', 'GENDER'] = 1
        data_gen['variable'] = 'Gender'
        data_gen.rename(columns={'GENDER':'value'}, inplace=True)
        data = pd.concat((data_age, data_gen), ignore_index=True)
        data['minute'] = 0
        self.events = pd.concat((self.events,data), ignore_index=True)

    def save_results_csv(self):
        self.meta_table.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'meta.csv'))
        self.input_real_time_standard.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'input_real_time.csv'))
        self.input_pre_admission.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'input_pre_admission.csv'))
        self.input_vasopressor_standard.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'input_vasopressor.csv'))
        self.output_real_time.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'output_real_time.csv'))
        self.output_pre_admission.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'output_pre_admission.csv'))
        self.chartevents.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'chartevents.csv'))
        self.labevents.to_csv(os.path.join(self.TO_SAVE_FOLDER_PATH,'labevents.csv'))
