% Main script for model 1 

%% Paths
% AT DTU

% Data
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\SampleData')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb02')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb04')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb05')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb07')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb10')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb10')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb13')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb14')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb16')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb20')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb21')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb22')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Spectral\freqs 32 new')

% Code
addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')

% AT HOME
% Data

% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\SampleData')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb02')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb04')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb05')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb07')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb10')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb10')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb13')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb14')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb16')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb20')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb21')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb22')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Spectral\freqs 32 new')
% 
% % Code
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Main')


%% Load data 
load('ersp_mat_02_32.mat')
Data_02 = ersp_mat_02_32;
seizure_02 = 2972:3053;
% load('EEG_clean_02_pre.mat')
% EEG_02_pre = EEG_clean_01_pre;
% Data_02_pre = EEG_02_pre.data;

load('ersp_mat_04_32.mat')
Data_04 = ersp_mat_04_32;
seizure_04 = 7804:7853;
% load('EEG_clean_04_pre.mat')
% EEG_04_pre = EEG_clean_04_pre;
% Data_04_pre = EEG_04_pre.data;

load('ersp_mat_05_32.mat')
Data_05 = ersp_mat_05_32;
seizure_05 = 417:532;
% load('EEG_clean_05_pre.mat')
% EEG_05_pre = EEG_clean_05_pre;
% Data_05_pre = EEG_05_pre.data;

load('ersp_mat_07_32.mat')
Data_07 = ersp_mat_07_32;
seizure_07 = 4920:5006;
% load('EEG_clean_07_pre.mat')
% EEG_07_pre = EEG_clean_07_pre;
% Data_07_pre = EEG_07_pre.data;

load('ersp_mat_10_32.mat')
Data_10 = ersp_mat_10_32;
seizure_10 = 6313:6348;
% load('EEG_clean_10_pre.mat')
% EEG_10_pre = EEG_clean_10_pre;
% Data_10_pre = EEG_10_pre.data;

load('ersp_mat_13_32.mat')
Data_13 = ersp_mat_13_32;
seizure_13 = 2077:2121;
% load('EEG_clean_13_pre.mat')
% EEG_13_pre = EEG_clean_13_pre;
% Data_13_pre = EEG_13_pre.data;

load('ersp_mat_14_32.mat')
Data_14 = ersp_mat_14_32;
seizure_14 = 1986:2000;
% load('EEG_clean_14_pre.mat')
% EEG_14_pre = EEG_clean_14_pre;
% Data_14_pre = EEG_14_pre.data;

load('ersp_mat_16_32.mat')
Data_16 = ersp_mat_16_32;
seizure_16 = 2290:2299;
% load('EEG_clean_16_pre.mat')
% EEG_16_pre = EEG_clean_16_pre;
% Data_16_pre = EEG_16_pre.data;

load('ersp_mat_20_32.mat')
Data_20 = ersp_mat_20_32;
seizure_20 = 94:123;
% load('EEG_clean_20_pre.mat')
% EEG_20_pre = EEG_clean_20_pre{1,1};
% Data_20_pre = EEG_20_pre.data;

load('ersp_mat_21_32.mat')
Data_21 = ersp_mat_21_32;
seizure_21 = 1288:1344;
% load('EEG_clean_21_pre.mat')
% EEG_21_pre = EEG_clean_21_pre;
% Data_21_pre = EEG_21_pre.data;

load('ersp_mat_22_32.mat')
Data_22 = ersp_mat_22_32;
seizure_22 = 3367:3425;
% load('EEG_clean_22_pre.mat')
% EEG_22_pre = EEG_clean_22_pre;
% Data_22_pre = EEG_22_pre.data;

load('freqs_32_07.mat')
freqs = freqs_32_07;

%% 
clear length_window Data title plot_name
length_window = 8;
Data = {Data_02, Data_04, Data_05, Data_07, Data_10, Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
% Data_pre = {Data_02_pre, Data_04_pre, Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
seizure = {seizure_02, seizure_04, seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
plot_name = {'Chb2', 'Chb4', 'Chb5', 'Chb7', 'Chb10', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};




%% FIRST APPROACH USING E_matrix for all channels
clear AreaROC_matrix Num_prin_matrix
size_train = 0.2;
length_window = 8;
var_target= [80 85 90 95 99];
for i = 2:length(Data)
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data{1,i}, false, [], [], length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix(i,:) =  AreaROC;
    Num_prin_matrix(i,:) = num_prin_vec;
end 


%% Print average Area under the curve

mean_AreaROC = mean(AreaROC_matrix);
std_AreaROC = std(AreaROC_matrix);

% Lets find 95% confidence interval
Z = 1.960;
n = length(Data);
conf_int = Z * std_AreaROC/sqrt(n);

lower_AreaROC = mean_AreaROC - conf_int;
upper_AreaROC = mean_AreaROC + conf_int;

mean_pc = mean(Num_prin_matrix);
std__pc = std(Num_prin_matrix);
conf_int = Z * std__pc/sqrt(n);

lower__pc =  mean_pc - conf_int;
upper__pc = mean_pc + conf_int;
xlabel_tick = [var_target 100];

figure()
yyaxis left 
h1 = plot(xlabel_tick,mean_AreaROC)
% hold on 
% plot(xlabel_tick,median_AreaROC)
hold on 
plot(xlabel_tick,lower_AreaROC, '--')
hold on 
plot(xlabel_tick,upper_AreaROC, '--')
ylabel('Area under the curve')
hold on
yyaxis right 
h2 = plot(xlabel_tick,mean_pc)
hold on
plot(xlabel_tick,lower__pc, '--')
hold on
plot(xlabel_tick,upper__pc, '--')
xlabel('% of variance explained')
ylabel('#Principal compenents')
legend([h1 h2], {'Mean area under the curve','Mean number of PC used (2d axis)'}, 'Location', 'northwest')


%% Learning curve
clear error_train error_test
xAxis_jump = 1
size_train = 0.1:0.1:0.90;
% size_train = 0.2;
var_target = 90;
for j = 1:length(Data)
    for i = 1:length(size_train)
        [E, E_train, var_explained] = E_matrix(Data{1,j}, size_train(i),length_window,'Plot', false ,[], 'Standardize', true);
        [E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target);
        
        [log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc); 
        error_train(i) =abs(mean(log_p_train));
        error_test(i) = abs(mean(log_p_test));
        
        Z = 1.960;
        n_train = length(log_p_train);
        std_train = std(log_p_train);
        conf_int_train(i) = Z * std_train/sqrt(n_train);
        
        n_test = length(log_p_test);
        std_test = std(log_p_test);
        conf_int_test(i) = Z * std_test/sqrt(n_test);
      
    end
    Model_1_errorPlot(error_train, error_test, conf_int_train, conf_int_test,size_train, xAxis_jump, title_plot{1,j}, plot_name{1,j})
end


%% Zero-One signal
for i =1:length(Data)
Data_test = Data{1,i};
Seizure_test = seizure{1,i};
SeizStart = Seizure_test(1);
SeizEnd = Seizure_test(end);
frac_worst = 0.05;
% frac_worst = C_ratio2(i);
var_target = 90;
[E, E_train, var_explained] = E_matrix(Data{1,i}, 0.2,length_window,'Plot', false ,[], 'Standardize', true);
[E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target); 
[log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc, 'Plot', true, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', true, [SeizStart SeizEnd], 10);
end





%% DIFFERENT APPROACH USING E_matrix for each channel

data = data_uni_rec;
size_train = 0.4;
var_target= 80;
[E_pc_mat, E_train_pc_mat] = channel_spectra(data, var_target, size_train);





