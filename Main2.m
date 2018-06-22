%% Paths
% AT DTU
% addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Lovisa')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Physio_clean_data')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Seizure_data')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb20')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\SampleData')
addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb16')
% % AT HOME
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Lovisa')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Physio_clean_data')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Seizure_data')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb20')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\SampleData')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Main')



%% Load data 
% load('SampleData_Clean')
% load('SampleData_Clean')
% load('EEG_clean_rec_09.mat');
% load('EEG_clean_20_pre.mat');
% load('EEG_clean_rec_09.mat');
% load('EEGICA_rec_09.mat');
load('EEG_clean_16_seiz.mat');
EEG_clean = EEG_clean_16_seiz;
% load('EEG_clean_20_post.mat');
% Define data
% EEG_clean = SampleData_Clean;
% EEG_clean = EEG_clean_20_pre{1,1};
load('EEG_clean_20_seiz_1Hz.mat');
% EEG_clean = EEG_clean_20_seiz_1Hz{1,1};

%% PCA and Energy 

% PCA_TrainTest
%   (
%   Data, 
%   Size_train, 
%   length_window, 
%   ** Optional **
%   'PC', 
%   true/false, 
%   number of PC used, 
%   'Plot', 
%   true/false, 
%   'Title'
%   );

Data = EEG_clean.data;
length_window = EEG_clean.srate;
[E E_train] = PCA_TrainTest(Data, 0.2, length_window, 'PC', true, 12, 'Plot', false, 'Patient 20: Seizure', false);
% NDATA = mat2gray(E);
% E = NDATA;
% figure()
% plot(E(1,:),E(2,:), '*')

%% Model 1

% Log likelihood
% This function calculates the log-likelihood for simple gaussian
% distribution. 

% Input 
%       E_train: Training data
%       E: The whole data set
% Optional
%       Plot: 
%           Plot_boleen = true/false
%           Plot_title =  'Title'
%           Fraction worst
%      Seizure:
%           Seizure_boleen = true/false
%           Seizure_time = [start:end] (sek)
% Output
%       log_p: log likelihood for all data
%       log_p_train: log liklehood for train data 
%       log_p_test: log liklehood for test data 
frac_worst = 0.03;
[log_p, log_p_train, log_p_test] = log_normal(E_train, E, 'Plot', true, 'Patient 20: Seizure', frac_worst, 'Seizure', true, [2290 2299], 10);

%% Model 1 validation

% This function calculates both the mean and norm of the log likelihood and
% plots up the error if that is wanted 

% Input:
%   Data
%   length_window
%   size_train
% Optional:
%   Plot
%   Plot_boleen
%   xAxis jump
% Output:
% error_train 
%   error_test 
%   error_train_norm 
%   error_test_norm
size_train = 0.1:0.1:0.9;
[error_train error_test error_train_norm error_test_norm] = Model_1_validation(Data, length_window,size_train, 21, 'Plot', true, 10);



%% Model 1 ROC plot 

Data = EEG_clean_16.data;
length_window_16 = EEG_clean_16.srate;
[E E_train] = PCA_TrainTest(Data, 0.2, length_window_16, 'PC', true, 12, 'Plot', false, 'Patient 20: Seizure');

Data_20 = EEG_clean_20.data;
length_window_20 = EEG_clean_20.srate;
[E_20 E_train_20] = PCA_TrainTest(Data_20, 0.2, length_window_20, 'PC', true, 12, 'Plot', false, 'Patient 20: Seizure');


seizure = 2290:2299;
seizure_20 = 94:123;


Frac = 0:5/3600:1;
% Frac = 0:0.01:1;


[FPR16, TPR16] = ROC_plot(E_train, E, Frac, seizure);
[FPR20, TPR20] = ROC_plot(E_train_20, E_20, Frac, seizure_20);
 
figure()
plot(FPR16,TPR16)
hold on
plot(FPR20, TPR20)
hold on
plot(FPR20, FPR20, '--')
xlabel('FPR')
ylabel('TPR')
legend('Patinet 16', 'Patient 20')
title('ROC plot')


figure()
plot(FPR16)
hold on 
plot(TPR16)
legend('FPR16', 'TPR16')

figure()
plot(FPR20)
hold on 
plot(TPR20)
legend('FPR20', 'TPR20')

%%  ROC curve different number of PC

num_pc = [80 85 90 95 99]; 
Data = EEG_clean_16_seiz.data;
length_window_16 = EEG_clean_16_seiz.srate;

seizure = 2290:2299;
Frac = 0:0.01:1;
    
FPR_matrix = zeros(length(num_pc), length(Frac));
TPR_matrix = zeros(length(num_pc), length(Frac));

AreaROC = zeros(1,length(num_pc));

for i = 1:length(num_pc)
    [E E_train] = PCA_TrainTest(Data, 0.2, length_window_16, 'PC', true, num_pc(i));

    [FPR16, TPR16] = ROC_plot(E_train, E, Frac, seizure);
    FPR_matrix(i,:) = FPR16;
    TPR_matrix(i,:) = TPR16;
    
    AreaROC(i) = trapz(FPR16,TPR16);
end 

figure()
for i = 1:length(num_pc)
    plot(FPR_matrix(i,:), TPR_matrix(i,:))
    hold on
end
legend('80%', '85%', '90%', '95%', '99%')
title('Patient 16')
xlabel('FPR')
ylabel('TPR')




%% Model 1 Q plot
[E E_train] = PCA_TrainTest(Data, 0.5, length_window, 'PC', true, 21, 'Plot', false, 'Patient 20: Seizure');
Emean = repmat(mean(E), 21,1);
EStd = repmat(std(E), 21,1);
E_norm = (E-Emean)./(EStd);
E_norm_train = E_norm(:,1800);
frac_worst = 0.05;

[log_p, log_p_train, log_p_test] = log_normal(E_train, E, 'Plot', false, 'Patient 20: Seizure', frac_worst, 'Seizure', true, [94 123], 10);
[log_p_norm, log_p_train_norm, log_p_test_norm] = log_normal(E_norm_train, E_norm, 'Plot', false, 'Patient 20: Seizure', frac_worst, 'Seizure', true, [94 123], 10);

t = floor(min(log_p)):0.01:ceil(max(log_p));
total_train_points = length(log_p_train);
total_test_points = length(log_p_test);
Q_train = zeros(1,length(t));
Q_test = zeros(1,length(t));
for i = 1:length(t)
    num_outlier_train = sum(log_p_train<t(i));
    Q_train(i) = num_outlier_train/total_train_points;
    num_outlier_test = sum(log_p_test<t(i));
    Q_test(i) = num_outlier_test/total_test_points;
end

t_norm = floor(min(log_p_norm)):0.01:ceil(max(log_p_norm));
total_train_points = length(log_p_train_norm);
total_test_points = length(log_p_test_norm);
Q_train_norm = zeros(1,length(t_norm));
Q_test_norm = zeros(1,length(t_norm));
for i = 1:length(t_norm)
    num_outlier_train = sum(log_p_train_norm<t_norm(i));
    Q_train_norm(i) = num_outlier_train/total_train_points;
    num_outlier_test = sum(log_p_test_norm<t_norm(i));
    Q_test_norm(i) = num_outlier_test/total_test_points;
end
figure()
plot(t,Q_train)
hold on 
plot(t,Q_test)
legend('Train', 'Test')

figure()
plot(t_norm,Q_train_norm)
hold on 
plot(t_norm,Q_test_norm)
legend('Train', 'Test')

%% Model 2

%% Optimal K 
[E E_train] = PCA_TrainTest(Data, 0.2, length_window, 'PC', true, [], 'Plot', false, ' Patient 20');

% NDATA = mat2gray(E);
% E = NDATA;
max_K = 40;
K_iter = 20;
nits=30;  
method=2; 
model2_optimalK(E(1:3,:), E_train(1:3,:), max_K, K_iter, nits, method);


%%
[E E_train] = PCA_TrainTest(Data, 0.2, length_window, 'PC', true, [], 'Plot', false, ' Patient 20');

% Normalize
% NDATA = mat2gray(E);
% E = NDATA;
% E_cut = E(1:10,:);
% E_train_cut = E_train(1:10, :)
K=15;                         % Number of clusters  
nits=30;                        % Number of EM iterations
method=2;                       % Method of initialization 1,2,3

model2(E, E_train, K, nits, method, 'Plot', false)

%% 
frac_worst = 0.03;

[E_sum_train prob_train]=gm_cost(E_train,y,sig2,prob_k);
[E_sum_test prob_test]=gm_cost(E_test,y,sig2,prob_k);

E_prob = [prob_train prob_test];

E_prob_sort = sort(E_prob);
ten_pro = E_prob_sort(floor(frac_worst*num_window));

[I,J] = find(ten_pro > E_prob);

ten_worst = zeros(1, num_window);
ten_worst(E_prob <= ten_pro) = 1;
ten_worst(E_prob > ten_pro) = 0;

figure()
subplot(2,1,1)
bar(ten_worst, 'b')
% hold on 
% bar(1:num_train_w, ten_worst(1:num_train_w), 'r')
title([num2str(frac_worst*100) '% worst for run ' run])
axis([0 num_window 0 1.2])
subplot(2,1,2)
hist(J,361)
axis([0 num_window 0 8])
title('Bin size: 10 sek MIXTURE')


%% Optimal K vs train/test

size_train = 0.1:0.1:0.9
max_K = 40;
K_iter = 20;
Error_train_matrix = [];
Error_test_matrix = [];
for i = 1:length(size_train)
    [E E_train] = PCA_TrainTest(Data, size_train(i), length_window, 'PC', false, [], 'Plot', fale, ' Patient 20');
    NDATA = mat2gray(E);
    [error_train_mean_vector, error_test_mean_vector] = model2_optimalK(E(:,:), E_train(:,:), max_K, K_iter, nits, method);
    Error_train_matrix(i,:) = error_train_mean_vector;
    Error_test_matrix(i,:) = error_test_mean_vector;
end

%% 
figure()
subplot(2,1,1)
plot(Error_train_matrix')
title('Train error') 
xlabel('K')
ylabel('mean log(p)')
legend('train size: 10%', 'train size: 20%','train size: 30%','train size: 40%','train size: 50%','train size: 60%','train size: 70%','train size: 80%','train size: 90%')

subplot(2,1,2)
plot(Error_test_matrix')
title('Test error') 
xlabel('K')
ylabel('mean log(p)')
legend('train size: 10%', 'train size: 20%','train size: 30%','train size: 40%','train size: 50%','train size: 60%','train size: 70%','train size: 80%','train size: 90%')

