function LoadData()
load('EEG_clean_02_seiz.mat')
EEG_02 = EEG_clean_02_seiz;
Data_02 = EEG_02.data;
seizure_02 = 2972:3053;
load('EEG_clean_02_pre.mat')
EEG_02_pre = EEG_clean_01_pre;
Data_02_pre = EEG_02_pre.data;

load('EEG_clean_04_seiz.mat')
EEG_04 = EEG_clean_04_seiz;
Data_04 = EEG_04.data;
seizure_04 = 7804:7853;
load('EEG_clean_04_pre.mat')
EEG_04_pre = EEG_clean_04_pre;
Data_04_pre = EEG_04_pre.data;

load('EEG_clean_05_seiz.mat')
EEG_05 = EEG_clean_05_seiz;
Data_05 = EEG_05.data;
seizure_05 = 417:532;
load('EEG_clean_05_pre.mat')
EEG_05_pre = EEG_clean_05_pre;
Data_05_pre = EEG_05_pre.data;

load('EEG_clean_07_seiz.mat')
EEG_07 = EEG_clean_07_seiz;
Data_07 = EEG_07.data;
seizure_07 = 4920:5006;
load('EEG_clean_07_pre.mat')
EEG_07_pre = EEG_clean_07_pre;
Data_07_pre = EEG_07_pre.data;

load('EEG_clean_10_seiz.mat')
EEG_10 = EEG_clean_10_seiz;
Data_10 = EEG_10.data;
seizure_10 = 6313:6348;
load('EEG_clean_10_pre.mat')
EEG_10_pre = EEG_clean_10_pre;
Data_10_pre = EEG_10_pre.data;

load('EEG_clean_13_seiz.mat')
EEG_13 = EEG_clean_13_seiz;
Data_13 = EEG_13.data;
seizure_13 = 2077:2121;
load('EEG_clean_13_pre.mat')
EEG_13_pre = EEG_clean_13_pre;
Data_13_pre = EEG_13_pre.data;

load('EEG_clean_14_seiz.mat')
EEG_14 = EEG_clean_14_seiz;
Data_14 = EEG_14.data;
seizure_14 = 1986:2000;
load('EEG_clean_14_pre.mat')
EEG_14_pre = EEG_clean_14_pre;
Data_14_pre = EEG_14_pre.data;

load('EEG_clean_16_seiz.mat')
EEG_16 = EEG_clean_16_seiz;
Data_16 = EEG_16.data;
seizure_16 = 2290:2299;
load('EEG_clean_16_pre.mat')
EEG_16_pre = EEG_clean_16_pre;
Data_16_pre = EEG_16_pre.data;

load('EEG_clean_20_seiz.mat')
EEG_20 = EEG_clean_20_seiz{1,1};
Data_20 = EEG_20.data;
seizure_20 = 94:123;
load('EEG_clean_20_pre.mat')
EEG_20_pre = EEG_clean_20_pre{1,1};
Data_20_pre = EEG_20_pre.data;

load('EEG_clean_21_seiz.mat')
EEG_21 = EEG_clean_21_seiz;
Data_21 = EEG_21.data;
seizure_21 = 1288:1344; 
load('EEG_clean_21_pre.mat')
EEG_21_pre = EEG_clean_21_pre;
Data_21_pre = EEG_21_pre.data;

load('EEG_clean_22_seiz.mat')
EEG_22 = EEG_clean_22_seiz;
Data_22 = EEG_22.data;
seizure_22 = 3367:3425;
load('EEG_clean_22_pre.mat')
EEG_22_pre = EEG_clean_22_pre;
Data_22_pre = EEG_22_pre.data;
end 