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
addpath('C:\Users\s161286\Dropbox\Master thesis\Data\smartphone\Clean')

% Code
addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')

% AT HOME
% Data
% % 
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\SampleData')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb02')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb04')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb05')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb07')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb10')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb10')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb13')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb14')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb16')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb20')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb21')
addpath('C:\Users\lovis\Dropbox\Master thesis\Data\Model1_data\Chb22')

% Code
addpath('C:\Users\lovis\Dropbox\Master thesis\Code\Main')
%% Load data 
load('EEG_clean_54.mat')
EEG_54 = EEG_clean_54;
Data_54 = EEG_54.data;

load('EEG_clean_55.mat')
EEG_55 = EEG_clean_55;
Data_55 = EEG_55.data;

load('EEG_clean_65.mat')
EEG_65 = EEG_clean_65;
Data_65 = EEG_65.data;

load('EEG_clean_68.mat')
EEG_68 = EEG_clean_68;
Data_68 = EEG_68.data;

load('EEG_clean_69.mat')
EEG_69 = EEG_clean_69;
Data_69 = EEG_69.data;

load('EEG_clean_70.mat')
EEG_70 = EEG_clean_70;
Data_70 = EEG_70.data;

load('EEG_clean_82.mat')
EEG_82 = EEG_clean_82;
Data_82 = EEG_82.data;

load('EEG_clean_83.mat')
EEG_83 = EEG_clean_83;
Data_83 = EEG_83.data;

load('EEG_clean_85.mat')
EEG_85 = EEG_clean_85;
Data_85 = EEG_85.data;

load('EEG_clean_86.mat')
EEG_86 = EEG_clean_86;
Data_86 = EEG_86.data;

load('EEG_clean_87.mat')
EEG_87 = EEG_clean_87;
Data_87 = EEG_87.data;


load('EEG_clean_90.mat')
EEG_90 = EEG_clean_90;
Data_90 = EEG_90.data;

load('EEG_clean_92.mat')
EEG_92 = EEG_clean_92;
Data_92 = EEG_92.data;

load('EEG_clean_93.mat')
EEG_93 = EEG_clean_93;
Data_93 = EEG_93.data;

load('EEG_clean_94.mat')
EEG_94 = EEG_clean_94;
Data_94 = EEG_94.data;

load('EEG_clean_97.mat')
EEG_97 = EEG_clean_97;
Data_97 = EEG_97.data;

load('EEG_clean_98.mat')
EEG_98 = EEG_clean_98;
Data_98 = EEG_98.data;

load('EEG_clean_99.mat')
EEG_99 = EEG_clean_99;
Data_99 = EEG_99.data;

load('EEG_clean_100.mat')
EEG_100 = EEG_clean_100;
Data_100 = EEG_100.data;

load('EEG_clean_101.mat')
EEG_101 = EEG_clean_101;
Data_101 = EEG_101.data;

load('EEG_clean_103.mat')
EEG_103 = EEG_clean_103;
Data_103 = EEG_103.data;

load('EEG_clean_104.mat')
EEG_104 = EEG_clean_104;
Data_104 = EEG_104.data;

load('EEG_clean_105.mat')
EEG_105 = EEG_clean_105;
Data_105 = EEG_105.data;
%% 
clear length_window Data title plot_name
length_window = EEG_54.srate;
Data_IED_raw = {Data_54,Data_65,Data_70,Data_83,Data_86, Data_90, Data_100};
Data_non_IED_raw = {Data_55,  Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};
Data_full = {Data_55,  Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105,Data_54,Data_65,Data_70,Data_83,Data_86, Data_90, Data_100}
title_plot = {'Patient 54', 'Patient 55', 'Patient 65','Patient 69', 'Patient 70', 'Patient 82', 'Patient 83', 'Patient 85', 'Patient 86', 'Patient 87','Patient 90', 'Patient 92', 'Patient 93', 'Patient 94', 'Patient 97', 'Patient 98', 'Patient 99', 'Patient 100', 'Patient 101', 'Patient 103', 'Patient 104', 'Patient 105'};
title_plot2 = {'Patient 54',  'Patient 65', 'Patient 70',  'Patient 83', 'Patient 86','Patient 90', 'Patient 100'};
title_plot3 = {'Patient 97', 'Patient 98', 'Patient 99', 'Patient 101', 'Patient 103', 'Patient 104', 'Patient 105'};
plot_name = {'S_55', 'S_69', 'S_82', 'S_85',  'S_87', 'S_92', 'S_93', 'S_94', 'S_97', 'S_98', 'S_99', 'S_101', 'S_103',  'S_104' , 'S_105', 'S_54', 'S_65','S_70','S_83','S_86', 'S_90', 'S_100'};
subjects = {'Subject 55', 'Subject 69', 'Subject 82', 'Subject 85',  'Subject 87', 'Subject 92', 'Subject 93', 'Subject 94', 'Subject 97', 'Subject 98', 'Subject 99', 'Subject 101', 'Subject 103',  'Subject 104' , 'Subject 105', 'Subject 54', 'Subject 65','Subject 70','Subject 83','Subject 86', 'Subject 90', 'Subject 100'}

title_non = {'Subject 97', 'Subject 98', 'Subject 99', 'Subject 101', 'Subject 103',  'Subject 104' , 'Subject 105'};
title_with = {'Subject 54','Subject 65','Subject 70','Subject 83','Subject 86', 'Subject 90', 'Subject 100'};
%% Clear out the first and last 60 sec in every recording 

clear_sec = 60;
sample_rate = 128;
num_points = clear_sec * sample_rate;

% Clean data full
for i = 1:length(Data_full)
    data = Data_full{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_full_clean{1,i} = data;
end 

% Clean data IED
for i = 1:length(Data_IED_raw)
    data = Data_IED_raw{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_WITH_clean{1,i} = data;
end 

% Clean data NON IED
for i = 1:length(Data_non_IED_raw)
    data = Data_non_IED_raw{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_NON_clean{1,i} = data;
end 



%% Devide all data in every channel by the MAD of every channel
Data_WITH_IED_MAD = {};
Data_NON_IED_MAD = {};
Data_full_cell = {};
MAD_matrix = [];
Data_full_MAD = {};
Data_full_STD = {};


for i = 1:length(Data_full_clean)
    data = Data_full_clean{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
    MAD_matrix(:,i) = MAD;   
    std_matrix(:,i) = std(data')';
    
    data_MAD = data./MAD;
    Data_full_MAD{1,i} = data_MAD;
    
    data_STD = data./std(data')';
    Data_full_STD{1,i} = data_STD;
    
    % Standardize MAD
    MAD_max = max(MAD);
    MAD_min = min(MAD);
%     MAD_matrix_std(:,i) = MAD./MAD_max;
    
end

for i = 1:14
    channel = MAD_matrix(i,:);
    
    MAD_matrix_std(i,:) = (channel - min(channel))/(max(channel) - min(channel));
%     MAD_matrix_std(i,:) = (channel)/(max(channel) );
end 

SUM_matrix_std = zeros(14,1);
SUM_matrix = zeros(14,1);
for i = 1:14
     channel = MAD_matrix_std(i,:);
    SUM_matrix_std = SUM_matrix_std + channel;
    
    
     channel = MAD_matrix(i,:);
    SUM_matrix = SUM_matrix + channel;
end 

figure()
for w = 1:1
    
    plot(1:length_non, SUM_matrix_std(w,1:length_non), '*')
    hold on
    plot(1+length_non:lenght_full,  SUM_matrix_std(w,length_non+1:end), '*')
    hold on 
    plot(repmat(min(SUM_matrix_std(w,length_non+1:end)), 1,lenght_full+3),'r--','LineWidth', 0.5)
    hold on 
    plot(repmat(max(SUM_matrix_std(w,1:length_non)), 1,lenght_full+3),'r--','LineWidth', 0.5)    
    xlabel('Subjects')
    ylabel('Sum of MAD')
%     title([' ' num2str(w)])
    axis([0 23 0 14])
    set(gca, 'FontSize', 14)
    
end

figure()
for w = 1:1
    
    plot(1:length_non, SUM_matrix(w,1:length_non), '*')
    hold on
    plot(1+length_non:lenght_full,  SUM_matrix(w,length_non+1:end), '*')
    hold on 
    plot(repmat(min(SUM_matrix(w,length_non+1:end)), 1,lenght_full+3),'r--','LineWidth', 1)
    hold on 
    plot(repmat(max(SUM_matrix(w,1:length_non)), 1,lenght_full+3),'r--','LineWidth', 1)    
    xlabel('Subjects')
    ylabel('Sum of MAD')
%     title([' ' num2str(])
%     axis([0 23 0 14])
    set(gca, 'FontSize', 14)
    
end

for i = 1:length(Data_WITH_clean)
    data = Data_WITH_clean{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
    MAD_matrix(:,i) = MAD;   
    
    data_MAD = data./MAD;
    Data_WITH_IED_MAD{1,i} = data_MAD;
    
end


for i = 1:length(Data_NON_clean)
    data = Data_NON_clean{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
    MAD_matrix(:,i) = MAD;   
    
    data_MAD = data./MAD;
    Data_NON_IED_MAD{1,i} = data_MAD;
    
end



%% Compute mean and std for each channel for each subject

for i = 1:length(Data_full_clean)
    mean_chan_clean(:,i) = mean(Data_full_clean{1,i},2);
    std_chan_clean(:,i) = std(Data_full_clean{1,i}')'./sqrt(length(Data_full_clean{1,i}));   
    
    mean_chan_MAD(:,i) = mean(Data_full_MAD{1,i},2);
    std_chan_MAD(:,i) = std(Data_full_MAD{1,i}')'./sqrt(length(Data_full_MAD{1,i}));   
    
    mean_chan_STD(:,i) = mean(Data_full_STD{1,i},2);
    std_chan_STD(:,i) = std(Data_full_STD{1,i}')'./sqrt(length(Data_full_STD{1,i}));  
    
end

%% 
length_non = 16;
length_full = length(Data_full);
for q = 1:14
h = figure()

subplot(2,1,1)
errorbar(1:length_non, mean_chan_clean(q,1:length_non),std_chan_clean(q,1:length_non))
hold on 
errorbar(length_non+1:length_full, mean_chan_clean(q,length_non+1:end),std_chan_clean(q,length_non+1:end))
% legend('Non IED', 'With IED', 'Location', 'northeastoutside')
title(['Before MAD  - Channel ' num2str(q)])
xlabel('Subject')
xticks(1:length_full)
xticklabels(subjects)
xtickangle(45)

subplot(2,1,2)
errorbar(1:length_non, mean_chan_MAD(q,1:length_non),std_chan_MAD(q,1:length_non))
hold on 
errorbar(length_non+1:length_full, mean_chan_MAD(q,length_non+1:end),std_chan_MAD(q,length_non+1:end))
title(['After MAD  - Channel ' num2str(q)])
xlabel('Subject')
% legend('Non IED', 'With IED', 'Location', 'northeastoutside')
xticks(1:length_full)
xticklabels(subjects)
xtickangle(45)

% subplot(2,1,2)
% errorbar(1:length_non, mean_chan_STD(q,1:length_non),std_chan_STD(q,1:length_non))
% hold on 
% errorbar(length_non+1:length_full, mean_chan_STD(q,length_non+1:end),std_chan_STD(q,length_non+1:end))
% title(['After STD  - Channel ' num2str(q)])
% xlabel('Subject')
% legend('Non IED', 'With IED')
% xticks(1:length_full)
% xticklabels(subjects)
% xtickangle(45)

 
  saveas(h, sprintf('Channel_MAD_%s', num2str(q)),'epsc')
end

%% 
figure()
length_non= 15;
lenght_full = 22;
  h =   figure()
for w = 1:14

     subplot(2,7,w)
    plot(1:length_non, MAD_matrix(w,1:length_non), '*')
    hold on
    plot(1+length_non:lenght_full,  MAD_matrix(w,length_non+1:end), '*')
    xlabel('subjects')
    title(['MAD - Channel ' num2str(w)])
%     axis([1 lenght_full 0 80])
    set(gca, 'FontSize', 14)
%     saveas(h, sprintf('MAD_matrix_chan_%s', num2str(w)),'epsc')
end


figure()
for w = 1:14
    subplot(2,7,w)
    plot(1:length_non, MAD_matrix_std(w,1:length_non), '*')
    hold on
    plot(1+length_non:lenght_full,  MAD_matrix_std(w,length_non+1:end), '*')
    xlabel('subjects')
    title(['Scaled - Channel ' num2str(w)])
    axis([0 23 0 1.2])
    set(gca, 'FontSize', 14)
end

%% Make the features for spectral
nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 128;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;
num_chan = 14;

% GET DATA WITH IED
 j = 1;
 Feature_test = {};
for k = 1:length(Data_WITH_IED_MAD)
    data = Data_WITH_IED_MAD{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);

    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;

        j = j+length_window;
    end
    
    Feature_test{1,k} = Feature_matrix;
    
end    

% GET DATA WITHOUT IED
  j = 1;
 Feature_cell_train = {};
for k = 1:length(Data_NON_IED_MAD)
    data = Data_NON_IED_MAD{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);

    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;

        j = j+length_window;
    end
    
    Feature_cell_train{1,k} = Feature_matrix;
  
end 


%% Split the data to train and test set

all_features = 84;
Data_train = [];

Num_sub_train = 8;
Num_with = 7;
Num_with = 15;


for i = 1:Num_sub_train
    Data_train = [Data_train Feature_cell_train{1,i}];
end

Data_test_non = [];
for i = Num_sub_train+1:Num_with
    Data_test_non = [Data_test_non Feature_cell_train{1,i}];
    Feature_test_non{1,i-Num_sub_train} = Feature_cell_train{1,i};
end

Data_test_with = [];
for i = 1:length(Feature_test)
    Data_test_with = [Data_test_with Feature_test{1,i}];
end

Feature_test_with = Feature_test;

%% Standardize data 

F_stand_train = {};
F_stand_test_non = {};
F_stand_test_with = {};

% Find the mean and standard deviation of the training set
mu_train = mean(Data_train,2);
std_train = std(Data_train')';

% Standardize the training set
F_stand_train{1,1} = (Data_train - mu_train)./std_train;

for k = 1:length(Feature_test_non)
    F_test_non = Feature_test_non{1,k};
    F_test_with = Feature_test_with{1,k};   

    F_stand_test_non{1,k} = (F_test_non - mu_train)./std_train;
    F_stand_test_with{1,k} = (F_test_with - mu_train)./std_train;
    
end 



%% Run PCA 
 % Run PCA on training data and applied to test data 
 
Y_train = {};
Y_test_non = {};
Y_test_with = {};
 
% PCA for training set and apply to test set
Train_data = F_stand_train{1,1};

[U, Y_trans, latent, tsquared, var_explained] = pca(Train_data', 'Centered', false);
Y_train{1,1} = Y_trans'; 

for k = 1:length(Feature_test_non)

    Test_data_non = F_stand_test_non{1,k};
    Test_data_with = F_stand_test_with{1,k};
    
    Y_test_non{1,k} = U'*Test_data_non;
    Y_test_with{1,k} = U'*Test_data_with;
    
end




%% MODELING STARTS 

% Define paramaters
e_max = 0.3;
alpha_0 = 0.4;
t_alpha = 6.6;
Q_t_max = 2*log(1./e_max);
num_prin = 14;
iter = 1;

% Get data
train = Y_train{1,1};
train = train(1:num_prin,:);

train_mean = mean(train,2); 
train_std = std(train')';
train = (train-train_mean)./train_std;

% Train model

[K_vec, y, sig2, prob_k] = model2_new_cooling(train,'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);  

y_old = y;
sig2_old = sig2;
%% Plot each subejct

% Novelty in IED data 
% y_full = y;
% sig2_full = sig2;
% [~, index] = sort(prob_k, 'descend');
% num_used = length(prob_k) - 2; %sum(prob_k > 0.0001)
% y_sort = y_full(index,:);
% y = y_sort(1:num_used,:);
% sig2_sort = sig2(index,:);
% sig2 = sig2_sort(1:num_used,:);

clear Mahalanobis_dist_vec_test Mahalanobis_dist_vec_train

figure()
for k = 1:length(Y_test_with)
    test = Y_test_with{1,k}(1:num_prin,:);
    test = (test-train_mean)./train_std;
    outlier_index_with = [];
    num_out = 0;
    clear Mahalanobis_dist_vec_test_with Mahalanobis_dist_vec_test_non
    for i = 1:length(test)
        x_t = test(:,i);
        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
        Mahalanobis_dist = min(Mahalanobis_dist);
        Mahalanobis_dist_vec_test_with(i) = Mahalanobis_dist;
        if Mahalanobis_dist > Q_t_max
            num_out = num_out + 1;
            outlier_index_with = [outlier_index_with i];
        end 
    end
    outlier_with = zeros(1,length(test));
    outlier_with(outlier_index_with) = 1;
    num_with(k) = sum(outlier_with);
    outlier_with_cell{1,k} = outlier_with;   
   

    subplot(4,2,k)
    plot(Mahalanobis_dist_vec_test_with)
    hold on 
    plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test_with)),'LineWidth', 1.5)
    xlabel('Time (s)', 'FontSize', 16)
    ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
    title(['With IED - Subject ' num2str(k)])
    yticks(Q_t_max)
    yticklabels('Q_t(\epsilon_{max})')
     axis([0 length(Mahalanobis_dist_vec_test_with) 0 3*Q_t_max])
end 

figure()
for k = 1:length(Y_test_non)
    clear Mahalanobis_dist_vec_test_with Mahalanobis_dist_vec_test_non
    test = Y_test_non{1,k}(1:num_prin,:);
    test = (test-train_mean)./train_std;
    outlier_index_non = [];
    for i = 1:length(test)
        x_t = test(:,i);
        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
        Mahalanobis_dist = min(Mahalanobis_dist);
        Mahalanobis_dist_vec_test_non(i) = Mahalanobis_dist;
        if Mahalanobis_dist > Q_t_max
            outlier_index_non = [outlier_index_non i];
        end 
    end 
    outlier_non = zeros(1,length(test));
    outlier_non(outlier_index_non) = 1;
    num_non(k) = sum(outlier_non);
    outlier_non_cell{1,k} = outlier_non;
    
    subplot(4,2,k)
    plot(Mahalanobis_dist_vec_test_non)
    hold on 
    plot(repmat(Q_t_max, 1,length(Mahalanobis_dist_vec_test_non)),'LineWidth', 1.5)
    xlabel('Time (s)', 'FontSize', 16)
    ylabel('Smallest Mahalanobis distance', 'FontSize', 14)
    title(['Normal Subject ' num2str(k)])
    yticks(Q_t_max)
    yticklabels('Q_t(\epsilon_{max})')
    axis([0 length(Mahalanobis_dist_vec_test_non) 0 3*Q_t_max])
end

%% Iterations 

% Train model q times
iter = 50
% y_cell = {};
% sig2_cell= {};
for q = 21:iter+20
    [K_vec, y, sig2] = model2_new_cooling(train,'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);  
    y_cell{1,q} = y;
    sig2_cell{1,q} = sig2;
end

%%
% Find outliers
Outlier_cell_with = {};

for k = 1:length(Y_test_with)
    test = Y_test_with{1,k}(1:num_prin,:);
    test = (test-train_mean)./train_std;
    outlier_matrix = [];
     Mahalanobis_WITH_MATRIX = [];
    
    for q = 1:iter+20
        y = y_cell{1,q};
        sig2 = sig2_cell{1,q}; 
        outlier_index_with = [];
        num_out = 0;
        clear Mahalanobis_dist_vec_test_with Mahalanobis_dist_vec_test_non
        for i = 1:length(test)
            x_t = test(:,i);
            Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
            Mahalanobis_dist = min(Mahalanobis_dist);
            Mahalanobis_dist_vec_test_with(i) = Mahalanobis_dist;
            if Mahalanobis_dist > Q_t_max
                num_out = num_out + 1;
                outlier_index_with = [outlier_index_with i];
            end 
        end
        outlier_with = zeros(1,length(test));
        outlier_with(outlier_index_with) = 1;
        outlier_matrix(q,:) = outlier_with;  
        Mahalanobis_WITH_MATRIX(q,:) = Mahalanobis_dist_vec_test_with;
    end
    Outlier_cell_with{1,k} = outlier_matrix;
    Mahalanobis_WITH_cell{1,k} = Mahalanobis_WITH_MATRIX;
end 

Outlier_cell_non = {};
for k = 1:length(Y_test_with)
    test = Y_test_non{1,k}(1:num_prin,:);
    test = (test-train_mean)./train_std;
    outlier_matrix = [];
    Mahalanobis_NON_MATRIX = [];
    for q = 1:iter+20
        y = y_cell{1,q};
        sig2 = sig2_cell{1,q}; 
        outlier_index_non = [];
        num_out = 0;
        clear Mahalanobis_dist_vec_test_with Mahalanobis_dist_vec_test_non
        for i = 1:length(test)
            x_t = test(:,i);
            Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
            Mahalanobis_dist = min(Mahalanobis_dist);
            Mahalanobis_dist_vec_test_non(i) = Mahalanobis_dist;
            if Mahalanobis_dist > Q_t_max
                num_out = num_out + 1;
                outlier_index_non = [outlier_index_non i];
            end 
        end
        outlier_non = zeros(1,length(test));
        outlier_non(outlier_index_non) = 1;
        outlier_matrix(q,:) = outlier_non;  
        Mahalanobis_NON_MATRIX(q,:) = Mahalanobis_dist_vec_test_non;
    end
    Outlier_cell_non{1,k} = outlier_matrix;
    Mahalanobis_NON_cell{1,k} = Mahalanobis_NON_MATRIX;
end 
    
%% Plot with development of outliers 
clear Growth_matrix_with Growth_matrix_non
for k = 1:length(Y_test_with)
    clear Growth_outlier
    Outlier_matrix_with = Outlier_cell_with{1,k};
    
    Growth_outlier(1) = sum(Outlier_matrix_with(1,:))
    old_outlier = Outlier_matrix_with(1,:);
    for i = 2:iter+20
        new_outlier = Outlier_matrix_with(i,:);
        change = old_outlier - new_outlier;
        Growth_outlier(i) = sum(change == -1);
        
        old_outlier = (new_outlier + old_outlier) > 0;
    end    
    Growth_matrix_with(k,:) = Growth_outlier;
    
    h = figure()
    plot(sum(Outlier_matrix_with))
    xlabel('Time')
    ylabel('Times novelty detected')
    title([title_with{1,k} ' - with IED'])
    set(gca, 'FontSize', 14)
    saveas(h, sprintf('Novelty_detected_with_%s', num2str(k)),'epsc')
end

for k = 1:length(Y_test_non)
    clear Growth_outlier
    Outlier_matrix_non = Outlier_cell_non{1,k};
    
    Growth_outlier(1) = sum(Outlier_matrix_non(1,:))
    old_outlier = Outlier_matrix_non(1,:);
    for i = 2:iter+20
        new_outlier = Outlier_matrix_non(i,:);
        change = old_outlier - new_outlier;
        Growth_outlier(i) = sum(change == -1);
        
        old_outlier = (new_outlier + old_outlier) > 0;
    end    
    Growth_matrix_non(k,:) = Growth_outlier;
    
    h = figure()
    plot(sum(Outlier_matrix_non))
    xlabel('Time')
    ylabel('Times novelty detected')
    title([title_non{1,k} ' - non IED'])
    set(gca, 'FontSize', 14)
    saveas(h, sprintf('Novelty_detected_non_%s', num2str(k)),'epsc')
    
end
figure()
subplot(2,1,1)
plot(Growth_matrix_with')
title('With IED')
xlabel('Iterations (s)')
ylabel('Num outliers')
subplot(2,1,2)
plot(Growth_matrix_non')
title('No IED')
xlabel('Iterations (s)')
ylabel('Num outliers')

sum(Growth_matrix_non,2)
sum(Growth_matrix_with,2)

figure()
plot(Growth_matrix_with(:,:)')
hold on
plot(Growth_matrix_non(:,:)')

figure()
Growth_matrix_total = [Growth_matrix_with;Growth_matrix_non];
avg_growth = mean(Growth_matrix_total);
plot(avg_growth)
xlabel('Number of runs (s)')
ylabel('Average number of new novel vectors')
set(gca, 'FontSize', 14)

%% LOG PLOT
for k = 1:length(Outlier_cell_with)
    outlier_matrix = Outlier_cell_with{1,k};
    outlier = zeros(1, length(outlier_matrix));
    outlier(sum(outlier_matrix)>5 ) = 1;
    num_outlier_with(k) = sum(outlier);
    not_outlier_with_1(k) = sum(sum(outlier_matrix)<=1 == sum(outlier_matrix)>0);

    h = figure()
    b1 = bar(outlier, 'b','EdgeColor', 'b')       
%     alpha(b1, 0.5)    
%     b1.EdgeAlpha = 0.10
    title([title_with{1,k} ' - with IED'])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)

%      saveas(h, sprintf('LogPlot_model2_SBS2_WITH_%s', num2str(k)),'epsc')
end

for k = 1:length(Outlier_cell_with)
    outlier_matrix = Outlier_cell_non{1,k};
    outlier = zeros(1, length(outlier_matrix));
    outlier(sum(outlier_matrix)>5 ) = 1;
    not_outlier_non_1(k) = sum(sum(outlier_matrix)<=1 == sum(outlier_matrix)>0);
    num_outlier_non(k) = sum(outlier);


    h = figure()
    b1 = bar(outlier, 'b','EdgeColor', 'b')       
%     alpha(b1, 0.5)    
%     b1.EdgeAlpha = 0.10
    title([title_non{1,k} ' - non IED '])
    axis([0 length(outlier) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)

%      saveas(h, sprintf('LogPlot_model2_SBS2_NON_s_%s', num2str(k)),'epsc')
end


%% Mean mahalanobis dist
figure()
for k = 1:length(Y_test_with)
    Distance = Mahalanobis_WITH_cell{1,k};
    mean_distiance = mean(Distance);
    std_distance = std(Distance);
    error = std_distance/size(Distance,1);
    lower = mean_distiance - error;
    upper = mean_distiance + error;
    
    mean_distance_with(k) = mean(mean_distiance);
    h = figure()
    plot(mean_distiance, 'b')
    hold on 
    plot(lower, 'b--')
    hold on 
    plot(upper, 'b--')
    hold on
    plot(repmat(Q_t_max, 1,length(mean_distiance)),'r','LineWidth', 1.5)
    xlabel('Time (s)', 'FontSize', 16)
    ylabel('D_M', 'FontSize', 14)
    title([title_with{1,k} ' - with IED'])
    yticks(Q_t_max)
    yticklabels('Q_t(\epsilon_{max})')
    axis([0 length(mean_distiance) 0 3*Q_t_max])    
    
     saveas(h, sprintf('Distance_WITH_s_%s', num2str(k)),'epsc')
end 

figure()
for k = 1:length(Y_test_non)
    Distance = Mahalanobis_NON_cell{1,k};
    mean_distiance = mean(Distance);
    std_distance = std(Distance);
    error = std_distance/size(Distance,1);
    lower = mean_distiance - error;
    upper = mean_distiance + error;
    
    mean_distance_non(k) = mean(mean_distiance);
   h = figure()
    plot(mean_distiance, 'b')
    hold on 
    plot(lower, 'b--')
    hold on 
    plot(upper, 'b--')
    hold on
    plot(repmat(Q_t_max, 1,length(mean_distiance)),'r','LineWidth', 1.5)
    xlabel('Time (s)', 'FontSize', 16)
    ylabel('D_M', 'FontSize', 14)
    title([title_non{1,k} ' - nonIED'])
    yticks(Q_t_max)
    yticklabels('Q_t(\epsilon_{max})')
    axis([0 length(mean_distiance) 0 3*Q_t_max])    
     saveas(h, sprintf('Distance_NON_s_%s', num2str(k)),'epsc')
end 











%% OLD 
% novelty_with = (mean(outlier_with_matrix)>0);
% novelty_non = (mean(outlier_non_matrix)>0);
% 
% sum(outlier_with_matrix)
% sum(outlier_non_matrix)

figure()
h_train = plot(train(1,:), train(2,:), 'm.')
hold on 
h_with = plot(test_with(1,:),test_with(2,:), 'b.')
hold on 
h_non = plot(test_non(1,:),test_non(2,:), 'r.')
hold on 
plot(y(:,1),y(:,2), '*')
for k=1:size(y,1)
   plot(y(k,1)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,2)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
end 
hold on 
h_non_novel = plot(test_non(1,novelty_non),test_non(2,novelty_non), 'ro')
hold on 
h_with_novel = plot(test_with(1,novelty_with),test_with(2,novelty_with), 'bo')

legend([h_train, h_with, h_non, h_non_novel , h_with_novel], {'Train', 'With IED', 'Not IED', 'Not IED novel', 'IED novel'})


%% TEST 
e_max =0.1:0.2:0.9;

[K_vec, y, sig2, prob_k] = model2_new_cooling(train,'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);



    clear Mahalanobis_dist_vec_test Mahalanobis_dist_vec_train
    for i = 1:length(test_with)
        x_t = test_with(:,i);
        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
        Mahalanobis_dist = min(Mahalanobis_dist);
        Mahalanobis_dist_vec_test_with(i) = Mahalanobis_dist;
        if Mahalanobis_dist > Q_t_max
            outlier_index_with = [outlier_index_with i];
        end 
    end 

    
    %% ALPHA_T
% Evalute paramters for GMM 

Dist_test_alphat_SBS2 = {};
Dist_train_alphat_SBS2 = {};
num_outlier_alphat_SBS2 = {};
num_K_alphat_SBS2 = {};
var_target = 95;
alpha_0 = 0.2:0.2:2;
tau_alpha = 5:0.2:7;
iter = 1;
e_max_opt = 0.3;
for d = 1:1%length(Data)
    d
    e_max = e_max_opt(d);
    var_explained = var_explained;
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 

    DataTrain = Y_train{1,d}(1:num_prin,:);
    [Dist_test, Dist_train, num_outlier_test, num_K] = Model2_alpha_t(DataTrain, alpha_0, tau_alpha, e_max, iter);
  
    Dist_test_alphat_SBS2{1,d} = Dist_test;
    Dist_train_alphat_SBS2{1,d} = Dist_train;
    num_outlier_alphat_SBS2{1,d} = num_outlier_test;
    num_K_alphat_SBS2{1,d} = num_K;
    
%     dist_opt = Dist_test_e_max./sum(Dist_test_e_max); 
%     novelty_opt = mean_num_outlier./sum(mean_num_outlier);
% 
%     [value index] = min(dist_opt + novelty_opt);
%     e_max_opt(d) = e_max_vec(index);
%     
%     h = figure()
%     yyaxis left 
%     h1 = plot(Dist_test_e_max)
%     hold on 
%     h2 = plot(Dist_train_e_max)
%     ylabel('Mean Mahalanobis distance (D)', 'FontSize', 16)
%     hold on 
%     yyaxis right
%     plot(mean_num_outlier)
%     legend([h1 h2], {'Test', 'Train'})
%     xticks(1:length(e_max_vec))
%     xticklabels(e_max_vec)
%     ylabel('Mean number of outlier (nov)', 'FontSize', 16)
%     xlabel('\epsilon_{max}', 'FontSize', 16)
%     title(title_plot{1,d})
%     set(gca, 'FontSize', 14)
% % 
%     saveas(h, sprintf('E_max_subject_%s', plot_name{1,d}),'epsc')
end