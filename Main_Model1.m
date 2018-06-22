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

% AT HOME
% Data
% % 
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

%% 
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = {Data_02, Data_05, Data_07, Data_10, Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
Data_pre = {Data_02_pre, Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
seizure = {seizure_02,  seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 2',  'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
plot_name = {'Chb2',  'Chb5', 'Chb7', 'Chb10', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};
% Data = {Data_02, Data_04, Data_05, Data_07, Data_10, Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
% Data_pre = {Data_02_pre, Data_04_pre, Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
% seizure = {seizure_02, seizure_04, seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
% title_plot = {'Patient 2', 'Patient 4', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
% plot_name = {'Chb2', 'Chb4', 'Chb5', 'Chb7', 'Chb10', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};

%% Print ROC plot for different number of variance explained
clear AreaROC_matrix AreaROC_matrix_stand Num_prin_matrix
var_target = 90;
for i =1:11
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC(Data{1,i}, false, [], [], length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix(i,:) =  AreaROC;
    Num_prin_matrix(i,:) = num_prin_vec;
    AreaROC_90(i) = AreaROC;
    num_prin_90(i) = num_prin_vec;
end
AreaROC_90 = AreaROC_90';
num_prin_90 = num_prin_90';
%% Print ROC plot for optimal variance explained for all the patient 
clear FPR_matrix TPR_matrix
var_target = 99;
for i = 1:length(title_plot)
    [FPR, TPR, AreaROC,num_prin_vec] = ROC_PC(Data{1,i},false,  [], [], length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
end

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
for i = 7:11
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
clear Q_train_matrix Q_test_matrix
size_train = 0.2;
length_window = 256;
t_cell = {};
Q_train_matrix = [];
Q_test_matrix = [];
var_target = 90;
j = 1;

t = -26000:0.1:5000;
for i = 1:11
    [E, E_train, var_explained] = E_matrix(Data{1,i}, size_train,length_window,'Plot', false ,[], 'Standardize', true);
    [E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target);
    [log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc); 
%     t = floor(min(log_p)):0.0001:ceil(max(log_p));
    min_log_p(i) = min(log_p);
    max_log_p(i) = max(log_p);
    [Q_train, Q_test] = Qplot(log_p, log_p_train, log_p_test, t);
    Q_train_matrix(i,:) = Q_train;
    Q_test_matrix(i,:) = Q_test;  
    j = j + 1;     
%     
%     figure()
%     plot(t,Q_train)
%     hold on
%     plot(t,Q_test)
%     legend('Train', 'Test')
%     xlabel('t')
%     ylabel('Q(t)')
%     title(title_plot{1,i})
end 
%%
clear Threshold
% Make 5% line 
Threshold(1:length(t)) = 0.05; 
figure()
subplot(2,1,1)
plot(t,Q_train_matrix')
hold on
plot(t,Threshold, '--')
title('Train')
xlabel('t')
ylabel('Q(t)')
axis([-26000 5000 0 1])
legend('p02', 'p04', 'p05','p07', 'p10', 'p13', 'p14', 'p16', 'p20', 'p21', 'p21')
subplot(2,1,2)
plot(t,Q_test_matrix')
hold on
plot(t,Threshold, '--')
title('Test')
axis([-26000 5000 0 1])

legend('p02', 'p04', 'p05','p07', 'p10', 'p13', 'p14', 'p16', 'p20', 'p21', 'p21')
xlabel('t')
ylabel('Q(t)')

%% Optimal threshold

% To find the optimal threshold we define objective function as the
% distance from the ROC curve to the point (0,1) that is 
%           C(t) = sqrt(1 - TPR(t)^2  + FPR(t)^2)

clear FPR_matrix TPR_matrix
var_target = 90;
length_window = 256;
for i = 1:length(title_plot)
    [FPR, TPR, AreaROC,num_prin_vec, f] = ROC_PC(Data{1,i}, false, [], [], length_window,var_target, seizure{1,i}, title_plot{1,i}, plot_name{1,i});
    FPR_matrix{i,:} =  FPR; 
    TPR_matrix{i,:} = TPR;
    C1 = sqrt((1 - TPR).^2 + FPR.^2);
    C1_min = min(C1);
    C1_min_index = find(C1 == C1_min);
    
    C_ratio(i) = C1_min_index/length(FPR);
    optimal_threshold(i) = f(C1_min_index);
    
     Sensitivity(i) = TPR(C1_min_index);
    Specificity(i) = 1 - FPR(C1_min_index);
    
    C2 = TPR-1.5*FPR;
    C2_min = max(C2);
    C2_min_index = find(C2 == C2_min);
    
    C_ratio2(i) = C2_min_index/length(FPR);
    optimal_threshold2(i) = f(C2_min_index);
    h = figure()
    plot(FPR,TPR)
    hold on 
    opt = plot(FPR(C1_min_index), TPR(C1_min_index), '*')
    hold on 
    opt2 = plot(FPR(C2_min_index), TPR(C2_min_index), '*')
    plot(FPR,FPR, '--')
    xlabel('FPR')
    ylabel('TPR')
    legend([opt opt2], {'Optimal threshold', 'Optimal threshold2'}, 'Location', 'southeast')
    title(title_plot{1,i})
    
%      saveas(h, sprintf('ROC_OptThres_%s', plot_name{1,i}),'epsc')
            
end

Sensitivity = Sensitivity';
Specificity = Specificity';
%% Learning curve
clear error_train error_test
xAxis_jump = 10;
size_train = 0.1:0.1:0.9;
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

