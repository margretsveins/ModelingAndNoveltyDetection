%% Paths
% AT DTU
% addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Lovisa')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Physio_clean_data')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Seizure_data')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb20')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\SampleData')
addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb16')
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb14')
% % AT HOME
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Lovisa')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Physio_clean_data')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Seizure_data')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb20')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Data\SampleData')
% addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Main')

%% Load data 
load('EEG_clean_14_seiz.mat');
EEG_14 = EEG_clean_14_seiz;
Data14 = EEG_14.data;

load('EEG_clean_16_seiz.mat');
EEG_16 = EEG_clean_16_seiz;
Data16 = EEG_16.data;

load('EEG_clean_20_seiz_1Hz.mat');
EEG_20 = EEG_clean_20_seiz_1Hz{1,1};
Data20 = EEG_20.data;

%%
Data = Data20;
length_window = EEG_20.srate;
[E_random E_train_random tf] = PCA_TrainTest_RANDOM(Data, 0.2, length_window, 'PC', true, 12, 'Plot', true, 'Patient 20: Seizure', false);
[E E_train] = PCA_TrainTest(Data, 0.2, length_window, 'PC', true, 12, 'Plot', true, 'Patient 20: Seizure', false);


%% 
frac_worst = 0.03;
[log_p_random, log_p_train_random, log_p_test_random ten_worst_random] = log_normal_RANDOM(E_train_random, E_random, tf,'Plot', true, 'Patient 20: Seizure', frac_worst, 'Seizure', true, [94 123], 10);
[log_p, log_p_train, log_p_test ten_worst] = log_normal(E_train, E ,'Plot', true, 'Patient 20: Seizure', frac_worst, 'Seizure', true, [94 123], 10);

True_posative = sum(ten_worst_random(94:123));
Condition_positive = length(94:123);
False_positive = sum(ten_worst_random) - True_posative;
Condition_negative = length(ten_worst_random)-Condition_positive;

TPR_random = True_posative/Condition_positive;
FPR_random = False_positive/Condition_negative;

True_posative = sum(ten_worst(94:123));
Condition_positive = length(94:123);
False_positive = sum(ten_worst) - True_posative;
Condition_negative = length(ten_worst)-Condition_positive;

TPR_NOTrandom = True_posative/Condition_positive;
FPR_NOTrandom = False_positive/Condition_negative;
;
%%
seizure_20 = 94:123;


Frac = 0:5/3600:1;
[FPR20_random, TPR20_random] = ROC_plot_RANDOM(E_train_random, E_random, Frac, seizure_20,tf);
[FPR20, TPR20] = ROC_plot(E_train, E, Frac, seizure_20);


%% 
figure()
plot(FPR20_random, TPR20_random)
hold on 
plot(FPR20, TPR20)
legend('random', 'not random')
title('ROC plot patient 20')
xlabel('FPR')
ylabel('TPR')

%% Model validation
size_train = 0.1:0.1:0.9;
Data = Data20;

 [error_train error_test] = Model_1_validation(Data, length_window,size_train, 12, 'Plot', true, 10);
[error_train_random error_test_random] = Model_1_validation_RANDOM(Data, length_window,size_train, 12, 'Plot', true, 10);

%% 
figure()
plot(error_train)
hold on 
plot(error_test)
hold on
plot(error_train_random)
hold on
plot(error_test_random)
legend('train', 'test', 'train random', 'test random')

%% 
error_train20 = error_train;
error_test20 = error_test;
error_train_random20 = error_train_random;
error_test_random20 = error_test_random;

%%
figure()
subplot(2,2,1)
plot(error_test14)
hold on
plot(error_test_random14)
hold on
plot(error_test16)
hold on
plot(error_test_random16)
hold on
plot(error_test20)
hold on
plot(error_test_random20)
hold off
legend('p14 normal', 'p14 random', 'p16 normal', 'p16 random', 'p20 normal', 'p20 random')
title('Test error')

subplot(2,2,2)
plot(error_test14)
hold on
plot(error_test_random14)
hold on
plot(error_train14)
hold on
plot(error_train_random14)
hold off
legend('p14 test normal', 'p14 test random', 'p14 train normal', 'p14 train random')
title('Patient 14')

subplot(2,2,3)
plot(error_test16)
hold on
plot(error_test_random16)
hold on
plot(error_train16)
hold on
plot(error_train_random16)
hold off
legend('p16 test normal', 'p16 test random', 'p16 train normal', 'p16 train random')
title('Patient 16')

subplot(2,2,4)
plot(error_test20)
hold on
plot(error_test_random20)
hold on
plot(error_train20)
hold on
plot(error_train_random20)
hold off
legend('p20 test normal', 'p20 test random', 'p20 train normal', 'p20 train random')
title('Patient 20')

