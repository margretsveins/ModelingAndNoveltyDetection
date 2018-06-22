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

% Code
addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')

% % AT HOME
% % Data
% 
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
% 
% % Code
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Main')


%% Load data 
load('EEG_rec_02_seiz.mat')
EEG_02 = EEG_rec_02_seiz;
Data_02 = EEG_02.data;
seizure_02 = 2972:3053;
load('EEG_rec_02_pre.mat')
EEG_02_pre = EEG_rec_01_pre;
Data_02_pre = EEG_02_pre.data;


load('EEG_rec_05_seiz.mat')
EEG_05 = EEG_rec_05_seiz;
Data_05 = EEG_05.data;
seizure_05 = 417:532;
load('EEG_rec_05_pre.mat')
EEG_05_pre = EEG_rec_05_pre;
Data_05_pre = EEG_05_pre.data;

load('EEG_rec_07_seiz.mat')
EEG_07 = EEG_rec_07_seiz;
Data_07 = EEG_07.data;
seizure_07 = 4920:5006;
load('EEG_rec_07_pre.mat')
EEG_07_pre = EEG_rec_07_pre;
Data_07_pre = EEG_07_pre.data;

load('EEG_rec_10_seiz.mat')
EEG_10 = EEG_rec_10_seiz;
Data_10 = EEG_10.data;
seizure_10 = 6313:6348;
load('EEG_rec_10_pre.mat')
EEG_10_pre = EEG_rec_10_pre;
Data_10_pre = EEG_10_pre.data;

load('EEG_rec_13_seiz.mat')
EEG_13 = EEG_rec_13_seiz;
Data_13 = EEG_13.data;
seizure_13 = 2077:2121;
load('EEG_rec_13_pre.mat')
EEG_13_pre = EEG_rec_13_pre;
Data_13_pre = EEG_13_pre.data;

load('EEG_rec_14_seiz.mat')
EEG_14 = EEG_rec_14_seiz;
Data_14 = EEG_14.data;
seizure_14 = 1986:2000;
load('EEG_rec_14_pre.mat')
EEG_14_pre = EEG_rec_14_pre;
Data_14_pre = EEG_14_pre.data;

load('EEG_rec_16_seiz.mat')
EEG_16 = EEG_rec_16_seiz;
Data_16 = EEG_16.data;
seizure_16 = 2290:2299;
load('EEG_rec_16_pre.mat')
EEG_16_pre = EEG_rec_16_pre;
Data_16_pre = EEG_16_pre.data;

load('EEG_rec_20_seiz.mat')
EEG_20 = EEG_rec_20_seiz{1,1};
Data_20 = EEG_20.data;
seizure_20 = 94:123;
load('EEG_rec_20_pre.mat')
EEG_20_pre = EEG_rec_20_pre{1,1};
Data_20_pre = EEG_20_pre.data;

load('EEG_rec_21_seiz.mat')
EEG_21 = EEG_rec_21_seiz;
Data_21 = EEG_21.data;
seizure_21 = 1288:1344;
load('EEG_rec_21_pre.mat')
EEG_21_pre = EEG_rec_21_pre;
Data_21_pre = EEG_21_pre.data;

load('EEG_rec_22_seiz.mat')
EEG_22 = EEG_rec_22_seiz;
Data_22 = EEG_22.data;
seizure_22 = 3367:3425;
load('EEG_rec_22_pre.mat')
EEG_22_pre = EEG_rec_22_pre;
Data_22_pre = EEG_22_pre.data;

%% 
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = {Data_02,  Data_05, Data_07, Data_10, Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
Data_pre = {Data_02_pre,  Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
seizure = {seizure_02,  seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 2',  'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
plot_name = {'Chb2', 'Chb5', 'Chb7', 'Chb10', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};

%% Print ROC plot for different number of variance explained
clear AreaROC_matrix AreaROC_matrix_stand Num_prin_matrix
var_target = 75:99;
for i = 1:length(title_plot)
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data{1,i}, length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix(i,:) =  AreaROC;
    Num_prin_matrix(i,:) = num_prin_vec;
end
%% Print ROC plot for optimal variance explained for all the patient 
clear FPR_matrix TPR_matrix
var_target = 90;
figure()
for i = 1:length(title_plot)
    [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC(Data{1,i}, length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
    
   plot(FPR,TPR)
   hold on
end
% legend([],{'Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'})
plot(FPR,FPR, '--')
xlabel('FPR')
ylabel('TPR')
title('ROC plot with 90% of variance explained')
%% 
figure()
for i = 1:6
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end 
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('90% of variance explained')
legend('Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Location', 'southeast')

figure()
for i = 6:11
    plot(FPR_matrix{i,:}, TPR_matrix{i,:})
    hold on
end
plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
xlabel('FPR')
ylabel('TPR')
title('90% of variance explained')
legend('Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22', 'Location', 'southeast')
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
xlabel_tick = 75:100;

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



%% Q - Plot
size_train = 0.2;
length_window = 256;
t_cell = {};
Q_train_matrix = [];
Q_test_matrix = [];
var_target = 90;
j = 1;
for i = 1:11
    [E, E_train, var_explained] = E_matrix(Data{1,i}, size_train,length_window,'Plot', false ,[], 'Standardize', true);
    [E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target);
    [log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc); 
    [t Q_train, Q_test] = Qplot(log_p, log_p_train, log_p_test);
    Q_train_matrix(i,:) = Q_train;
    Q_test_matrix(i,:) = Q_test;  
    j = j + 1;     
    
    figure()
    plot(t,Q_train)
    hold on
    plot(t,Q_test)
    legend('Train', 'Test')
    xlabel('t')
    ylabel('Q(t)')
    title(title_plot{1,i})
end 

% figure()
% subplot(2,1,1)
% plot(Q_train_matrix')
% title('Train')
% legend('p02', 'p04', 'p05','p07', 'p10', 'p13', 'p14', 'p16', 'p20', 'p21', 'p21')
% subplot(2,1,2)
% plot(Q_test_matrix')
% title('Test')
% legend('p02', 'p04', 'p05','p07', 'p10', 'p13', 'p14', 'p16', 'p20', 'p21', 'p21')

%% Learning curve for 