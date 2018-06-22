% Run for one patient at a time and test K
%% Load data 
load('EEG_clean_02_seiz.mat')
EEG_02 = EEG_clean_02_seiz;
Data_02 = EEG_02.data;
seizure_02 = 2972:3053;
load('EEG_clean_02_pre.mat')
EEG_02_pre = EEG_clean_01_pre;
Data_02_pre = EEG_02_pre.data;

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


%% Get featuresets
nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 256;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;
num_chan = 21;

% GET TEST DATA
 j = 1;
 Feature1_cell_test = {};
 Feature2_cell_test = {};
 Feature2__test_norm_cell = {};
 PSE_test_cell = {};
for k = 1:length(Data)
    data = Data{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    Feature_matrix_norm = zeros(5*num_chan,num_window);
    PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        Feature_matrix_norm(:,i) = E_normal;
        PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature2_cell_test{1,k} = Feature_matrix;
    Feature2__test_norm_cell{1,k} = Feature_matrix_norm;
    PSE_test_cell{1,k} = PSE_matrix; 
    Feature1_cell_test{1,k} = GetFeatures1(data, length_window);
    
end    

% GET CLEAN DATA (TRAIN DATA)
  j = 1;
 Feature1_cell_train = {};
 Feature2_cell_train = {};
 Feature2_norm_cell_train = {};
 PSE_cell_train = {};
for k = 1:length(Data_pre)
    data = Data_pre{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    Feature_matrix_norm = zeros(5*num_chan,num_window);
    PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        Feature_matrix_norm(:,i) = E_normal;
        PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature2_cell_train{1,k} = Feature_matrix;
    Feature2_norm_cell_train{1,k} = Feature_matrix_norm;
    PSE_cell_train{1,k} = PSE_matrix;   
    Feature1_cell_train{1,k} = GetFeatures1(data, length_window);
end

%% Standardize data 
F_1_stand_test = {};
F_1_stand_train = {};
F_2_stand_test = {};
F_2_stand_train = {};

for k = 1:length(Feature1_cell_train)
    Feature_1_train = Feature1_cell_train{1,k};
    Feature_1_test = Feature1_cell_test{1,k};
    Feature_2_train = Feature2_cell_train{1,k};
    Feature_2_test = Feature2_cell_test{1,k};  
    
    mu_1 = mean(Feature_1_train,2);
    mu_2 = mean(Feature_2_train,2);
    
    std_1 = std(Feature_1_train')';
    std_2 = std(Feature_2_train')';
    
    F_1_stand_test{1,k} = (Feature_1_test - mu_1)./std_1;
    F_1_stand_train{1,k} = (Feature_1_train - mu_1)./std_1;
    F_2_stand_test{1,k} = (Feature_2_test - mu_2)./std_2;
    F_2_stand_train{1,k} = (Feature_2_train - mu_2)./std_2;
    
end 


%% Run PCA 
 % Run PCA on training data and applied to test data 
 
 Y_train_F1 = {};
 Y_test_F1 = {};
 Y_train_F2 = {};
 Y_test_F2 = {};
for k = 1:length(Feature1_cell_train)
    
    % Featurset 1 
    Train_data_F1 = F_1_stand_train{1,k};
    Test_data_F1 = F_1_stand_test{1,k} ;
    [U_F1, Y_trans_F1, latent_F1, tsquared_F1, var_explained] = pca(Train_data_F1', 'Centered', false);
    Y_train_F1{1,k} = Y_trans_F1'; 
    Y_test_F1{1,k} = U_F1'*Test_data_F1;
    var_explained_F1{1,k} = var_explained;
    
    % Featurset 2 
    Train_data_F2 = F_2_stand_train{1,k};
    Test_data_F2 = F_2_stand_test{1,k} ;
    [U_F2, Y_trans_F2, latent_F2, tsquared_F2, var_explained] = pca(Train_data_F2', 'Centered', false);
    Y_train_F2{1,k} = Y_trans_F2'; 
    Y_test_F2{1,k} = U_F2'*Test_data_F2;
    var_explained_F2{1,k} = var_explained;
    
end

%%

E_test = Y_test_F2{1,2}(1:3,:);
E_train = Y_train_F2{1,2}(1:3,:);

train_mean = mean(E_train,2); 
train_std = std(E_train')';
E_train = (E_train-train_mean)./train_std;
E_test = (E_test-train_mean)./train_std;

E = [E_train E_test];

max_K = 20;
K_iter = 10;
nits = 20;
method = 1;

[error_train_mean_vector, error_test_mean_vector] = model2_optimalK(E, E_train, max_K, K_iter, nits, method)







