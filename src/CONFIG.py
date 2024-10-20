#######################
## file path configs ##
#######################

DATA_RAW_PATH = '../mimic3-all' #can change to ../demo_data or ../mimic3-carevue
PREPROCESSED_DATA_FOLDER_PATH = '../data/preprocessed' #define the folder you want to save preprocessed results
                                                #change to '' if you dont want to save
PREPROCESSED_DATA_FILE_NAME = 'mimic_iii.pkl'


#####################
## feature configs ##
#####################

# It is used when preprocessing the raw data.

#TODO: sanity check 
#1. unit check
#2. outlier removal
#3. check duplicates

DITEM_ID_IN_SCOPE = {#### Vital Signs ####
                    'blood_pressure_diastolic':{'ids':[8368,220051,8555],'label':'dbp'},
                    'blood_pressure_Systolic':{'ids':[51,6701,220050],'label':'sbp'},
                    'blood_pressure_Systolic':{'ids':[52,6702,220052],'label':'sbp'},
                    'GCS_eye':{'ids':[184,220739],'label':'gcs_eye'},
                    'GCS_motor':{'ids':[454,223901],'label':'gcs_motor'},
                    'GCS_verbal':{'ids':[723,223900],'label':'gcs_verbal'},
                    'heart_rate':{'ids':[211, 220045],'label':'hr'},
                    'spontaneous_respiratory_rate':{'ids':[614,651,224689,224422],'label':'srr'},
                    'respiratory_rate':{'ids':[618,3603,220210],'label':'rr'},
                    #'total_respiratory_rate':{'ids':[615,224690],'label':'trr'},
                    'temperature':{'ids':[3655, 677, 676, 223762],'label':'temperature'}, #TODO: need to check if 223761, 678, 679, 3654 are the same measuresments in F units
                    #### Demographics ####
                    'weight':{'ids':[224639,763],'label':'weight'},
                    'height':{'ids':[1394, 226707],'label':'height'},#TODO: need to checkk 226730
                    #### Ventilation ####
                    'FiO2':{'ids':[3420, 223835, 3422, 189, 727, 190],'label':'FiO2'},#TODO: convert percentage to fraction
                    #### Lab measures ####
                    'blood_glucose':{'ids':[225664, 1529, 811, 807, 3745, 50809],'label':'bg'},#TODO: both chartevents and labevents record this item
                    }

VITAL_IN_SCOPE = {}


DLABITEM_ID_IN_SCOPE = {'bilirubin_total':{'ids':[50885],'label':'bilirubin'},
                        'Oxygen_saturation':{'ids':[834, 50817, 8498, 220227, 646, 220277],'label':'O2_saturation'},#TODO: both chartevents and labevents
                        'Potassium':{'ids':[50971, 50822],'label':'Potassium'},
                        'Chloride':{'ids':[50902, 50806],'label':'Chloride'},
                        'Magnesium':{'ids':[50960],'label':'Magnesium'},
                        'Calcium Total':{'ids':[50893],'label':'Calcium_tot'},
                        'Calcium Free':{'ids':[50808],'label':'Calcium_free'},
                        'white_blood_cell_count':{'ids':[51301, 51300],'label':'wbc'},
                        'Hemoglobin':{'ids':[51222, 50811],'label':'Hemoglobin'},#Hemoglobin is a protein in red blood cells that carries oxygen.
                        'Urea Nitrogen':{'ids':[51006],'label':'BUN'},
                        'Creatinine Blood': {'ids':[50912],'label':'creatinine_blood'},
                        'Creatinine Urine': {'ids':[51082],'label':'creatinine_urine'},
                        'Bicarbonate':{'ids':[50882, 50803],'label':'bicarbonate'},
                        'Albumin':{'ids':[50862],'label':'albumin'}, #it's a thing made by liver
                        'Lactate':{'ids':[50813],'label':'lactate'}, # Lactate is a bi-product constantly produced in the body during normal metabolism and exercise
                        'Platelet Count':{'ids':[51265],'label':'platelet_count'}, #Platelets are cells that help your blood clot
                        'PT':{'ids':[51274],'label':'PT'},
                        'PTT':{'ids':[51275],'label':'PTT'},
                        'Total CO2':{'ids':[50804],'label':'CO2_tot'},
                        'PO2':{'ids':[50821],'label':'PO2'},#PO2 (partial pressure of oxygen) reflects the amount of oxygen gas dissolved in the blood
                        'PCO2':{'ids':[50818],'label':'PCO2'},
                        'Base Excess':{'ids':[50802],'label':'base_excess'},
                        'pH Urine':{'ids':[51491, 51094, 220734, 1495, 1880, 1352, 6754, 7262],'label':'urine_ph'},
                        'pH Blood':{'ids':[50820],'label':'blood_pd'},
                        }
#from https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_Data_extract_MIMIC3_140219.ipynb
INPUT_TOTAL_EQUIVALENT_VOL_COEF = {30176:0.25,
                                   30315:0.25,
                                   30161:0.3,
                                   227531:2.75,
                                   30143:3,
                                   225161:3,
                                   30009:5,
                                   220862:5,
                                   30030:6.6,
                                   220995:6.6,
                                   227533:6.6,
                                   228341:8
}
for i in [30020,30015,225823,30321,30186,30211, 30353,42742,42244,225159]:
    INPUT_TOTAL_EQUIVALENT_VOL_COEF[i] = 0.5

PRE_ADM_FLUID_INTAKE = [30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,227071,227072]

VASOPRESSORS_ITEMIDS = [30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289, 222315, 221662, 30043, 30307]
VASOPRESSORS_RATE_CONVERSION = {
    (30120, 'mcg/kg/min'): 1, (221906, 'mcg/kg/min'): 1, (30047, 'mcg/kg/min'): 1,  # norad
    (30120, 'mcgkgmin'): 1, (221906, 'mcgkgmin'): 1, (30047, 'mcgkgmin'): 1,  # norad from carevue
    (30120, 'mcg/min'): 1/80, (221906, 'mcg/min'): 1/80, (30047, 'mcg/min'): 1/80,  # norad
    (30120, 'mcgmin'): 1/80, (221906, 'mcgmin'): 1/80, (30047, 'mcgmin'): 1/80,  # norad from carevue
    (30119, 'mcg/kg/min'): 1, (221289, 'mcg/kg/min'): 1,  # epi
    (30119, 'mcgkgmin'): 1, (221289, 'mcgkgmin'): 1, 
    (30119, 'mcg/min'): 1/80, (221289, 'mcg/min'): 1/80,  # epi
    (30119, 'mcgmin'): 1/80, (221289, 'mcgmin'): 1/80,
    (30051, 'units/min'): 5, (222315, 'units/min'): 5,  # vasopressin
    (30051, 'Umin'): 5, (222315, 'Umin'): 5, 
    (30051, 'units/hour'): 5/60, (222315, 'units/hour'): 5/60,  # vasopressin
    (30051, 'Uhr'): 5/60, (222315, 'Uhr'): 5/60,
    (30051, '> 0.2'): 5/60,  # vasopressin (rate > 0.2)
    (222315, '> 0.2'): 5/60, 
    (30128, 'mcg/kg/min'): 0.45, (221749, 'mcg/kg/min'): 0.45, (30127, 'mcg/kg/min'): 0.45,  # phenyl
    (30128, 'mcgkgmin'): 0.45, (221749, 'mcgkgmin'): 0.45, (30127, 'mcgkgmin'): 0.45,
    (30128, 'mcg/min'): 0.45/80, (221749, 'mcg/min'): 0.45/80, (30127, 'mcg/min'): 0.45/80,  # phenyl
    (30128, 'mcgmin'): 0.45/80, (221749, 'mcgmin'): 0.45/80, (30127, 'mcgmin'): 0.45/80,
    (221662, 'mcg/kg/min'): 0.01, (30043, 'mcg/kg/min'): 0.01, (30307, 'mcg/kg/min'): 0.01,  # dopa
    (221662, 'mcgkgmin'): 0.01, (30043, 'mcgkgmin'): 0.01, (30307, 'mcgkgmin'): 0.01,
    (221662, 'mcg/min'): 0.01/80, (30043, 'mcg/min'): 0.01/80, (30307, 'mcg/min'): 0.01/80,  # dopa
    (221662, 'mcgmin'): 0.01/80, (30043, 'mcgmin'): 0.01/80, (30307, 'mcgmin'): 0.01/80
}

OUTPUT_ITEMID = {'pre_admission':{'ids':[40060,226633],'label':'pre_admission'},
                 'real_time':{'ids':[40055,226559],'label':'urine(Foley)'}} #urine out Foley

#Glasgow coma scale(GCS):
#The GCS is scored between 3 and 15, 3 being the worst and 15 the best. 
#It is composed of three parameters: best eye response (E), best verbal response (V), and best motor response (M)
#E2V3M4 results in a GCS score of 9

#The fraction of inspired oxygen (FiO2) is the concentration of oxygen in the gas mixture. 
#The gas mixture at room air has a fraction of inspired oxygen of 21%, meaning that the concentration of oxygen at room air is 21%.

#bilirubin (bil-ih-ROO-bin) is a yellowish pigment that is made during the breakdown of red blood cells.

#Urea nitrogen is a waste product that your kidneys remove from your blood. 
#Higher than normal BUN levels may be a sign that your kidneys aren't working well. 

#Base excess is defined as the amount of strong acid that must be added to each liter of fully oxygenated blood 
#to return the pH to 7.40 at a temperature of 37°C and a pCO2 of 40 mmHg (5.3 kPa), 
#while a base deficit (ie. a negative base excess) is defined by the amount of strong base that must be added.
