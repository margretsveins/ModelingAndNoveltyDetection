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
load('EEG_clean_02_seiz.mat')
EEG_02 = EEG_clean_02_seiz;
Data_02 = EEG_02.data;
seizure_02 = 2972:3053;
load('EEG_clean_02_pre.mat')
EEG_02_pre = EEG_clean_01_pre;
Data_02_pre = EEG_02_pre.data;
load('EEG_clean_02_post.mat')
EEG_02_post = EEG_clean_02_post;
Data_02_post = EEG_02_post.data;


load('EEG_clean_05_seiz.mat')
EEG_05 = EEG_clean_05_seiz;
Data_05 = EEG_05.data;
seizure_05 = 417:532;
load('EEG_clean_05_pre.mat')
EEG_05_pre = EEG_clean_05_pre;
Data_05_pre = EEG_05_pre.data;
load('EEG_clean_05_post.mat')
EEG_05_post = EEG_clean_05_post;
Data_05_post = EEG_05_post.data;

load('EEG_clean_07_seiz.mat')
EEG_07 = EEG_clean_07_seiz;
Data_07 = EEG_07.data;
seizure_07 = 4920:5006;
load('EEG_clean_07_pre.mat')
EEG_07_pre = EEG_clean_07_pre;
Data_07_pre = EEG_07_pre.data;
load('EEG_clean_07_post.mat')
EEG_07_post = EEG_clean_07_post;
Data_07_post = EEG_07_post.data;

load('EEG_clean_10_seiz.mat')
EEG_10 = EEG_clean_10_seiz;
Data_10 = EEG_10.data;
seizure_10 = 6313:6348;
load('EEG_clean_10_pre.mat')
EEG_10_pre = EEG_clean_10_pre;
Data_10_pre = EEG_10_pre.data;
load('EEG_clean_10_post.mat')
EEG_10_post = EEG_clean_10_post;
Data_10_post = EEG_10_post.data;

load('EEG_clean_13_seiz.mat')
EEG_13 = EEG_clean_13_seiz;
Data_13 = EEG_13.data;
seizure_13 = 2077:2121;
load('EEG_clean_13_pre.mat')
EEG_13_pre = EEG_clean_13_pre;
Data_13_pre = EEG_13_pre.data;
load('EEG_clean_13_post.mat')
EEG_13_post = EEG_clean_13_post;
Data_13_post = EEG_13_post.data;

load('EEG_clean_14_seiz.mat')
EEG_14 = EEG_clean_14_seiz;
Data_14 = EEG_14.data;
seizure_14 = 1986:2000;
load('EEG_clean_14_pre.mat')
EEG_14_pre = EEG_clean_14_pre;
Data_14_pre = EEG_14_pre.data;
load('EEG_clean_14_post.mat')
EEG_14_post = EEG_clean_14_post;
Data_14_post = EEG_14_post.data;

load('EEG_clean_16_seiz.mat')
EEG_16 = EEG_clean_16_seiz;
Data_16 = EEG_16.data;
seizure_16 = 2290:2299;
load('EEG_clean_16_pre.mat')
EEG_16_pre = EEG_clean_16_pre;
Data_16_pre = EEG_16_pre.data;
load('EEG_clean_16_post.mat')
EEG_16_post = EEG_clean_16_post;
Data_16_post = EEG_16_post.data;

load('EEG_clean_20_seiz.mat')
EEG_20 = EEG_clean_20_seiz{1,1};
Data_20 = EEG_20.data;
seizure_20 = 94:123;
load('EEG_clean_20_pre.mat')
EEG_20_pre = EEG_clean_20_pre{1,1};
Data_20_pre = EEG_20_pre.data;
load('EEG_clean_20_post.mat')
EEG_20_post = EEG_clean_20_post{1,1};
Data_20_post = EEG_20_post.data;

load('EEG_clean_21_seiz.mat')
EEG_21 = EEG_clean_21_seiz;
Data_21 = EEG_21.data;
seizure_21 = 1288:1344; 
load('EEG_clean_21_pre.mat')
EEG_21_pre = EEG_clean_21_pre;
Data_21_pre = EEG_21_pre.data;
load('EEG_clean_21_post.mat')
EEG_21_post = EEG_clean_21_post;
Data_21_post = EEG_21_post.data;

load('EEG_clean_22_seiz.mat')
EEG_22 = EEG_clean_22_seiz;
Data_22 = EEG_22.data;
seizure_22 = 3367:3425;
load('EEG_clean_22_pre.mat')
EEG_22_pre = EEG_clean_22_pre;
Data_22_pre = EEG_22_pre.data;
load('EEG_clean_22_post.mat')
EEG_22_post = EEG_clean_22_post;
Data_22_post = EEG_22_post.data;

%% 
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = {Data_02, Data_05, Data_07, Data_10, Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
Data_pre = {Data_02_pre, Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
Data_post = {Data_02_post, Data_05_post, Data_07_post, Data_10_post, Data_13_post, Data_14_post, Data_16_post, Data_20_post, Data_21_post,Data_22_post};
seizure = {seizure_02,  seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 2',  'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
plot_name = {'Chb2',  'Chb5', 'Chb7', 'Chb10', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};

%% WITH OUT 10
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = {Data_02, Data_05, Data_07,  Data_13, Data_14, Data_16, Data_20, Data_21,Data_22};
Data_pre = {Data_02_pre, Data_05_pre, Data_07_pre,  Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
Data_post = {Data_02_post, Data_05_post, Data_07_post,  Data_13_post, Data_14_post, Data_16_post, Data_20_post, Data_21_post,Data_22_post};
seizure = {seizure_02,  seizure_05, seizure_07,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 2',  'Patient 5', 'Patient 7','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22'};
plot_name = {'Chb2',  'Chb5', 'Chb7', 'Chb13','Chb14', 'Chb16', 'Chb20', 'Chb21', 'Chb22'};

%% WITH OUT 02, 05, 10, 21 and 22
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = {Data_07,  Data_13, Data_14, Data_16, Data_20};
Data_pre = {Data_07_pre,  Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre};
Data_post = {Data_07_post,  Data_13_post, Data_14_post, Data_16_post, Data_20_post};
seizure = {seizure_07,seizure_13, seizure_14, seizure_16, seizure_20};
title_plot = {'Patient 7','Patient 13', 'Patient 14', 'Patient 16', 'Patient 20'};
plot_name = {'Chb7', 'Chb13','Chb14', 'Chb16', 'Chb20'};
%% WITH OUT 02, 05, 07, 13, 14, 10, 21 and 22
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = { Data_16, Data_20};
Data_pre = { Data_16_pre, Data_20_pre};
Data_post = {Data_16_post, Data_20_post};
seizure = {seizure_16, seizure_20};
title_plot = {'Patient 16', 'Patient 20'};
plot_name = { 'Chb16', 'Chb20'};

%% WITH OUT 02, 05, 07, 10 ,16, 20, 21 and 22
clear length_window Data title plot_name
length_window = EEG_20.srate;
Data = { Data_13, Data_14};
Data_pre = { Data_13_pre, Data_14_pre};
Data_post = { Data_13_post, Data_14_post};
seizure = {seizure_13, seizure_14};
title_plot = {'Patient 13', 'Patient 14'};
plot_name = { 'Chb13','Chb14'};
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
for k = 1:length(Data)
    data = Data{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);    
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        
        E = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        j = j+length_window;
    end    
    Feature2_cell_test{1,k} = Feature_matrix;   
    Feature1_cell_test{1,k} = GetFeatures1(data, length_window);    
end    

% GET CLEAN DATA (TRAIN DATA)
  j = 1;
 Feature1_cell_train = {};
 Feature2_cell_train = {}; 
for k = 1:length(Data_pre)
    data = Data_pre{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);    
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        
        % Get the features
        E= getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;       
        j = j+length_window;
    end    
    Feature2_cell_train{1,k} = Feature_matrix;   
    Feature1_cell_train{1,k} = GetFeatures1(data, length_window);
end

% GET CLEAN TEST DATA (POST DATA)
 j = 1;
 Feature1_cell_test_clean = {};
 Feature2_cell_test_clean = {};
for k = 1:length(Data)
    data = Data_post{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);    
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        
        E = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        j = j+length_window;
    end    
    Feature2_cell_test_clean{1,k} = Feature_matrix;   
    Feature1_cell_test_clean{1,k} = GetFeatures1(data, length_window);    
end  

%% Standardize data 
F_1_stand_test = {};
F_1_stand_train = {};
F_2_stand_test = {};
F_2_stand_test_clean = {};
F_2_stand_train = {};


for k = 1:length(Feature1_cell_train)
    Feature_1_train = Feature1_cell_train{1,k};
    Feature_1_test = Feature1_cell_test{1,k};
    Feature_2_train = Feature2_cell_train{1,k};
    Feature_2_test = Feature2_cell_test{1,k};  
    Feature_2_test_clean = Feature2_cell_test_clean{1,k};
    
    mu_1 = mean(Feature_1_train,2);
    mu_2 = mean(Feature_2_train,2);
    
    std_1 = std(Feature_1_train')';
    std_2 = std(Feature_2_train')';
    
    F_1_stand_test{1,k} = (Feature_1_test - mu_1)./std_1;
    F_1_stand_train{1,k} = (Feature_1_train - mu_1)./std_1;
    F_2_stand_test{1,k} = (Feature_2_test - mu_2)./std_2;
    F_2_stand_test_clean{1,k} = (Feature_2_test_clean - mu_2)./std_2;
    F_2_stand_train{1,k} = (Feature_2_train - mu_2)./std_2;
    
end 


%% Run PCA 
 % Run PCA on training data and applied to test data 
 
 Y_train_F1 = {};
 Y_test_F1 = {};
 Y_train_F2 = {};
 Y_test_F2 = {};
 Y_test_clean_F2 = {};
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
    Test_clean_data_F2 = F_2_stand_test_clean{1,k} ;
    [U_F2, Y_trans_F2, latent_F2, tsquared_F2, var_explained] = pca(Train_data_F2', 'Centered', false);
    Y_train_F2{1,k} = Y_trans_F2'; 
    Y_test_F2{1,k} = U_F2'*Test_data_F2;
    Y_test_clean_F2{1,k} = U_F2'*Test_clean_data_F2;
    var_explained_F2{1,k} = var_explained;
    
end


%% Find optimal variance explained.
clear     AreaROC_matrix_F1 Num_prin_matrix_F1  AreaROC_matrix_F2 Num_prin_matrix_F2
var_target = 75:99; %[80 85 90 95 99];
for i = 1:length(Feature1_cell_train)
    % Get seizure
    seizure_before = seizure{1,i};
    
    % Adjust seizure if we take window not equal 1 sek
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
%     Get info for feature 1
    Data_train_F1 = Y_train_F1{1,i};
    Data_test_F1 = Y_test_F1{1,i};
    Data_F1 = [Data_train_F1 Data_test_F1];
    size_train = length(Data_train_F1)/(length(Data_train_F1) + length(Data_test_F1)); 
    
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC_spectral(Data_F1, var_explained_F1{1,i}, length_window,size_train,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix_F1(i,:) =  AreaROC;
    Num_prin_matrix_F1(i,:) = num_prin_vec;
    
      % Get info for feature 2
    Data_train_F2 = Y_train_F2{1,i};
    Data_test_F2 = Y_test_F2{1,i};
    Data_F2 = [Data_train_F2 Data_test_F2];
    size_train = length(Data_train_F2)/(length(Data_train_F2) + length(Data_test_F2)); 
    
    [FPR_matrix, TPR_matrix, AreaROC,num_prin_vec] = ROC_PC_spectral(Data_F2, var_explained_F2{1,i}, length_window,size_train,var_target, seizure_new, title_plot{1,i}, plot_name{1,i});
    AreaROC_matrix_F2(i,:) =  AreaROC;
    Num_prin_matrix_F2(i,:) = num_prin_vec;
end 

AreaROC_matrix_Feature = {AreaROC_matrix_F1, AreaROC_matrix_F2};
Num_prin_matrix_Feature = {Num_prin_matrix_F1,Num_prin_matrix_F2};
%% Print average Area under the curve
for j = 1:2
    AreaROC_matrix = AreaROC_matrix_Feature{1,j};
    Num_prin_matrix = Num_prin_matrix_Feature{1,j};
    
    mean_AreaROC = mean(AreaROC_matrix);
    std_AreaROC = std(AreaROC_matrix);

    % Lets find 95% confidence interval
    Z = 1;
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
    ylim([0 1])
    ylabel('Area under the curve')
    hold on
    yyaxis right 
    
    h2 = plot(xlabel_tick,mean_pc)
    hold on
    plot(xlabel_tick,lower__pc, '--')
    hold on
    plot(xlabel_tick,upper__pc, '--')
    ylim([0 140])
    xlabel('% of variance explained')
    ylabel('#Principal compenents')
    legend([h1 h2], {'Mean area under the curve','Mean number of PC used (2d axis)'}, 'Location', 'northwest')
    title(['Feature set ' num2str(j)])
end


%% Print ROC plot for optimal variance explained for all the patient 
clear FPR_matrix TPR_matrix
var_target_F1 = 90;
var_target_F2 = 77;
for i = 1:length(Feature1_cell_test)
     % Get seizure
    seizure_before = seizure{1,i};
    
    % Adjust seizure if we take window not equal 1 sek
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
%     Get info for feature 1
    Data_train_F1 = Y_train_F1{1,i};
    Data_test_F1 = Y_test_F1{1,i};
    Data_F1 = [Data_train_F1 Data_test_F1];
    size_train = length(Data_train_F1)/(length(Data_train_F1) + length(Data_test_F1)); 
    
    [FPR_F1, TPR_F1, AreaROC,num_prin_vec_F1] = ROC_PC_spectral(Data_F1, var_explained_F1{1,i}, length_window,size_train,var_target_F1, seizure_new, title_plot{1,i}, plot_name{1,i});
    FPR_matrix_F1{i,:} =  FPR_F1; 
    TPR_matrix_F1{i,:} = TPR_F1;    
    num_prin_F1(i,1) = num_prin_vec_F1;
    AreaROC_F1(i,1) = AreaROC;
    
    % Get info for feature 2
    Data_train_F2 = Y_train_F2{1,i};
    Data_test_F2 = Y_test_F2{1,i};
    Data_F2 = [Data_train_F2 Data_test_F2];
    size_train = length(Data_train_F2)/(length(Data_train_F2) + length(Data_test_F2));     
    
    [FPR_F2, TPR_F2, AreaROC,num_prin_vec_F2] = ROC_PC_spectral(Data_F2, var_explained_F2{1,i}, length_window,size_train,var_target_F2, seizure_new, title_plot{1,i}, plot_name{1,i});
    FPR_matrix_F2{i,:} =  FPR_F2; 
    TPR_matrix_F2{i,:} = TPR_F2;
    num_prin_F2(i,1) = num_prin_vec_F2;
    AreaROC_F2(i,1) = AreaROC;
end
FPR_matrix_Feat = {FPR_matrix_F1,FPR_matrix_F2};
TPR_matrix_Feat = {TPR_matrix_F1,TPR_matrix_F2};

%% Print ROC plot for optimal variance explained for all the patient 
for j = 1:2
    FPR_matrix = FPR_matrix_Feat{1,j};
    TPR_matrix = TPR_matrix_Feat{1,j};
    h1 = figure()
    for i = 1:5
        plot(FPR_matrix{i,:}, TPR_matrix{i,:})
        hold on
    end 
    plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
    xlabel('FPR')
    ylabel('TPR')
    title(['Feature set ' num2str(j)])
    legend('Patient 2', 'Patient 5', 'Patient 7', 'Patient 10','Patient 13', 'Location', 'southeast')
%     saveas(h1, sprintf('ROC_all_optPC_first_F%d', j),'epsc')

    h2 = figure()
    for i = 6:10
        plot(FPR_matrix{i,:}, TPR_matrix{i,:})
        hold on
    end
    plot(FPR_matrix{i,:},FPR_matrix{i,:},'--')
    xlabel('FPR')
    ylabel('TPR')
    title(['Feature set ' num2str(j)])
    legend('Patient 14', 'Patient 16', 'Patient 20', 'Patient 21', 'Patient 22', 'Location', 'southeast')
%     saveas(h2, sprintf('ROC_all_optPC_sec_F%d', j),'epsc')
end

%% Log normal
var_target_F1 = 90;
var_target_F2 = 77;

frac_worst = 0.01;   
for i =1:length(Feature1_cell_test)
    
    % Get seizure
    Seizure_test = seizure{1,i};
    SeizStart = floor(Seizure_test(1)/num_sec);
    SeizEnd = ceil(Seizure_test(end)/num_sec);
    
    % Get info for feature 1       
    % Get correct number of features
    var_explained = var_explained_F1{1,i};   
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target_F1
        num_prin = num_prin + 1;
    end 
    
    Data_train_F1 = Y_train_F1{1,i};
    Data_train_F1 = Data_train_F1(1:num_prin,:);
    Data_test_F1 = Y_test_F1{1,i};
    Data_test_F1 = Data_test_F1(1:num_prin,:);
    Data_F1 = [Data_train_F1 Data_test_F1];        
    num_window = size(Data_F1,2);
    [log_p, log_p_train, log_p_test] = log_normal_clean(Data_train_F1, Data_F1, 'Plot', true, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', true, [SeizStart SeizEnd], 10);
    log_p_F1{1,i} = log_p;
    log_p_train_F1{1,i} = log_p_train;
    log_p_test_F1{1,i} = log_p_test;
    
    % Get info for feature 2     
    % Get correct number of features
    var_explained = var_explained_F2{1,i};   
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target_F2
        num_prin = num_prin + 1;
    end 
    
    Data_train_F2 = Y_train_F2{1,i};
    Data_train_F2 = Data_train_F2(1:num_prin,:);
    Data_test_F2 = Y_test_F2{1,i};
    Data_test_F2 = Data_test_F2(1:num_prin,:);
    Data_F2 = [Data_train_F2 Data_test_F2];        
    num_window = size(Data_F2,2);
    [log_p, log_p_train, log_p_test, ] = log_normal_clean(Data_train_F2, Data_F2, 'Plot', true, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', true, [SeizStart SeizEnd], 10);
    log_p_F2{1,i} = log_p;
    log_p_train_F2{1,i} = log_p_train;
    log_p_test_F2{1,i} = log_p_test;
end

%% Q plot 

% Figure of Q plot when standardized - DO NOT USE
jump = 0.05;
figure()
for i = 1:length(log_p_train_F1)
    % Feature set 1 
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
    mu_train_F1 = mean(F1_log_p_train);
    std_train_F1 = std(F1_log_p_train);   
    mu_test_F1 = mean(F1_log_p_train);
    std_test_F1 = std(F1_log_p_train);  
    F1_log_p_train = (F1_log_p_train-mu_train_F1)/std_train_F1;
    F1_log_p_test = (F1_log_p_test-mu_test_F1)/std_test_F1;

    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F1_log_p_train, F1_log_p_test, jump);
    subplot(2,2,1)
    hold on 
    plot(t_train,Q_train)
    hold on
    plot([-30 5], [0.01 0.01], 'r--')
    title('F1 train - standardized')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-30 5 0 1])
    subplot(2,2,2)
    hold on 
    plot(t_test,Q_test)
    hold on
    plot([-30 5], [0.01 0.01], 'r--')
    title('F1 test - standardized')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-30 5 0 1])

    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};
    
    mu_train_F2 = mean(F2_log_p_train);
    std_train_F2 = std(F2_log_p_train);   
    F2_log_p_train = (F2_log_p_train-mu_train_F2)/std_train_F2;
    F2_log_p_test = (F2_log_p_test-mu_train_F2)/std_train_F2;
    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F2_log_p_train, F2_log_p_test, jump);
    subplot(2,2,3)
    hold on 
    plot(t_train,Q_train)
    hold on
    plot([-30 5], [0.01 0.01], 'r--')
    title('F2 train - standardized')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-30 5 0 1])
    subplot(2,2,4)
    hold on
    plot(t_test,Q_test)
    hold on
    plot([-30 5], [0.01 0.01], 'r--')
    title('F2 test - standardized')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-30 5 0 1])
end 
%%
% Figure of Q plot not standardized
jump = 0.05;
figure()
threshold = -15;
Q_train_F1 = {};
Q_test_F1 = {};
t_train_F1 = {};
t_test_F1 = {};

Q_train_F2 = {};
Q_test_F2 = {};
t_train_F2 = {};
t_test_F2 = {};

for i = 1:length(log_p_train_F1)
    % Feature set 1 
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
    F1_train_thres(i) = sum(F1_log_p_train<threshold)/length(F1_log_p_train);
    F1_test_thres(i) = sum(F1_log_p_test<threshold)/length(F1_log_p_test);

    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F1_log_p_train, F1_log_p_test, jump);
    Q_train_F1{1,i} = Q_train;
    Q_test_F1{1,i} = Q_test;
    t_train_F1{1,i} = t_train;
    t_test_F1{1,i} = t_test;

    
    subplot(2,2,1)
    hold on 
    plot(t_train,Q_train)
    title('F1 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5])
    subplot(2,2,2)
    hold on 
    plot(t_test,Q_test)
    title('F1 test')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5])

%     figure()
%     plot(t_train,Q_train)
%     hold on 
%     plot(t_test,Q_test)
%     title('F1 test')
%     xlabel('\theta')
%     ylabel('Q(\theta)')
%     legend('train','test')

    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};

    F2_train_thres(i) = sum(F2_log_p_train<threshold)/length(F2_log_p_train);
    F2_test_thres(i) = sum(F2_log_p_test<threshold)/length(F2_log_p_test);
    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F2_log_p_train, F2_log_p_test, jump);
    Q_train_F2{1,i} = Q_train;
    Q_test_F2{1,i} = Q_test;
    t_train_F2{1,i} = t_train;
    t_test_F2{1,i} = t_test;
    
    subplot(2,2,3)
    hold on 
    plot(t_train,Q_train)
    title('F2 train')
    xlabel('\theta')
    ylabel('Q(\theta)')
    axis([-150 5 0 0.5])
    subplot(2,2,4)
    hold on
    plot(t_test,Q_test)
    title('F2 test')
    xlabel('\theta')
    ylabel('Q(\theta)')  
    axis([-150 5 0 0.5])
%     figure()
%     plot(t_train,Q_train)
%     hold on
%     plot(t_test,Q_test)
%     title('F2 test')
%     xlabel('\theta')
%     ylabel('Q(\theta)')   
%     legend('train','test')
end 


%% Single Threshold decided from Q plot for all sets

Q_opt_F1 = 0.02; %0.005:0.005:0.03;
Q_opt_F2 = 0.02; %0.005:0.005:0.03;

t_test_thres_F1 = zeros(length(Q_opt_F1),length(Q_test_F1));
t_test_thres_F2 = zeros(length(Q_opt_F2),length(Q_test_F2));
Q_test_F1_thres = zeros(length(Q_opt_F1),length(Q_test_F1));
Q_test_F2_thres = zeros(length(Q_opt_F2),length(Q_test_F2));
for j= 1:length(Q_opt_F1)
    for i = 1:length(Q_train_F1)

        Q_train_temp =  find(Q_train_F1{1,i} > Q_opt_F1(j));
        Q_train_thres = Q_train_temp(1);

        t_train_F1_thres = t_train_F1{1,i}(Q_train_thres);

        t_test_temp = find(t_test_F1{1,i} > t_train_F1_thres);
        t_test_thres = t_test_temp(1);
        t_test_thres_F1(j,i) = t_test_F1{1,i}(t_test_thres);

        Q_test_F1_thres(j,i) = Q_test_F1{1,i}(t_test_thres);
    end

    
    for i = 1:length(Q_train_F2)

        Q_train_temp =  find(Q_train_F2{1,i} > Q_opt_F2(j));
        Q_train_thres = Q_train_temp(1);

        t_train_F2_thres = t_train_F2{1,i}(Q_train_thres);

        t_test_temp = find(t_test_F2{1,i} > t_train_F2_thres);
        t_test_thres = t_test_temp(1);
        t_test_thres_F2(j,i) = t_test_F2{1,i}(t_test_thres);

        Q_test_F2_thres(j,i) = Q_test_F2{1,i}(t_test_thres);
    end

end

%% Log normal
var_target_F1 = 90;
var_target_F2 = 77;

sensitivity_F1 = zeros(length(Q_opt_F1),length(Feature1_cell_test));
specificity_F1 = zeros(length(Q_opt_F1),length(Feature1_cell_test));
sensitivity_F2 = zeros(length(Q_opt_F1),length(Feature1_cell_test));
specificity_F2 = zeros(length(Q_opt_F1),length(Feature1_cell_test));

frac_worst = 0.01;   

for j = 1:length(Q_opt_F1)
    for i =1:length(Feature1_cell_test)

        % Get seizure
        Seizure_test = seizure{1,i};
        SeizStart = floor(Seizure_test(1)/num_sec);
        SeizEnd = ceil(Seizure_test(end)/num_sec);

        % Get info for feature 1       
        % Get correct number of features
%         var_explained = var_explained_F1{1,i};   
%         num_prin = 1;
%         while sum(var_explained(1:num_prin)) < var_target_F1
%             num_prin = num_prin + 1;
%         end 
% 
%         Data_train_F1 = Y_train_F1{1,i};
%         Data_train_F1 = Data_train_F1(1:num_prin,:);
%         Data_test_F1 = Y_test_F1{1,i};
%         Data_test_F1 = Data_test_F1(1:num_prin,:);
%         Data_F1 = [Data_train_F1 Data_test_F1];        
%         num_window = size(Data_F1,2);
%         [log_p, log_p_train, log_p_test, ten_worst] = log_normal_clean(Data_train_F1, Data_F1, 'Plot', true, title_plot{1,i}, plot_name{1,i}, Q_test_F1_thres(j,i), 'Seizure', true, [SeizStart SeizEnd], 10);
%         true_positive = sum(ten_worst(Seizure_test));
% 
%         False_postive = sum(ten_worst)-true_positive;
%         num_seizure = length(Seizure_test);
%         Condition_negative = length(Data_test_F1)-num_seizure;
% 
%         TPR = true_positive/num_seizure;
%         FPR = False_postive/Condition_negative;
% 
%         sensitivity_F1(j,i) = TPR;
%         specificity_F1(j,i) = 1-FPR;
% 
%         log_p_F1{1,i} = log_p;
%         log_p_train_F1{1,i} = log_p_train;
%         log_p_test_F1{1,i} = log_p_test;

        % Get info for feature 2     
        % Get correct number of features
        var_explained = var_explained_F2{1,i};   
        num_prin = 1;
        while sum(var_explained(1:num_prin)) < var_target_F2
            num_prin = num_prin + 1;
        end 

        Data_train_F2 = Y_train_F2{1,i};
        Data_train_F2 = Data_train_F2(1:num_prin,:);
        Data_test_F2 = Y_test_F2{1,i};
        Data_test_F2 = Data_test_F2(1:num_prin,:);
        Data_F2 = [Data_train_F2 Data_test_F2];        
        num_window = size(Data_F2,2);
        [log_p, log_p_train, log_p_test, ten_worst] = log_normal_clean(Data_train_F2, Data_F2, 'Plot', true, title_plot{1,i}, plot_name{1,i}, Q_test_F2_thres(j,i), 'Seizure', true, [SeizStart SeizEnd], 10);

        true_positive = sum(ten_worst(Seizure_test));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_test);
        Condition_negative = length(Data_test_F2)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;

        sensitivity_F2(j,i) = TPR;
        specificity_F2(j,i) = 1-FPR;

        log_p_F2{1,i} = log_p;
        log_p_train_F2{1,i} = log_p_train;
        log_p_test_F2{1,i} = log_p_test;
    end


end

%%
figure()
plot(sensitivity_F2)
hold on
plot(specificity_F2)
hold on
plot(Q_test_F2_thres)
title('Feature set 2')
legend('Sensitivity','Specificity')

figure()
plot(sensitivity_F1)
hold on
plot(specificity_F1)
hold on
plot(Q_test_F1_thres)
title('Feature set 1')
legend('Sensitivity','Specificity')


figure()
plot(Q_opt_F2(1:5), mean(sensitivity_F2(1:5,:)'))
hold on
plot(Q_opt_F2(1:5), mean(specificity_F2(1:5,:)'))
% hold on
% plot(Q_test_F2_thres)
ylim([0 1])
title('Feature set 2')
legend('Sensitivity','Specificity')

figure()
subplot(2,1,1)
bar(Q_opt_F1, sensitivity_F1)
title('Sensitivity - Feature set 1')
hold on
subplot(2,1,2)
bar(Q_opt_F1, specificity_F1)
ylim([0 1])
title('Specificity - Feature set 1')

figure()
subplot(2,1,1)
bar(Q_opt_F2, sensitivity_F2)
title('Sensitivity - Feature set 2')
hold on
subplot(2,1,2)
bar(Q_opt_F2, specificity_F2)
ylim([0 1])
title('Specificity - Feature set 2')

%%
sens_prec_1_F1= mean(sensitivity_F1(2,:))
sens_prec_2_F1= mean(sensitivity_F1(4,:))

spec_prec_1_F1 = mean(specificity_F1(2,:))
spec_prec_2_F1 = mean(specificity_F1(4,:))

sens_prec_1_F2= mean(sensitivity_F1(2,:))
sens_prec_2_F2= mean(sensitivity_F1(4,:))

spec_prec_1_F2 = mean(specificity_F1(2,:))
spec_prec_2_F2 = mean(specificity_F1(4,:))

%%
figure()
plot(F1_train_thres)
hold on 
plot(F1_test_thres)
hold on 
plot(F2_train_thres)
hold on 
plot(F2_test_thres)
legend('F1 train','F1 test','F2 train','F2 test')

for i = 1:length(log_p_train_F1)
    Seizure_time = seizure{1,i};
    
    % Feature set 1
    F1_log_p_train = log_p_train_F1{1,i};
    F1_log_p_test = log_p_test_F1{1,i};
    
    % Standardize
%     mu_train_F1 = mean(F1_log_p_train);
%     std_train_F1 = std(F1_log_p_train);    
%     F1_log_p_test = (F1_log_p_test-mu_train_F1)/std_train_F1;
    
    % Plot
    log_plot(F1_log_p_test, Threshold_F1, title_plot{1,i},Seizure_time)
       
    
    % Feature set 2
    F2_log_p_train = log_p_train_F2{1,i};
    F2_log_p_test = log_p_test_F2{1,i};
    
    % Standardize
%     mu_train_F2 = mean(F2_log_p_train);
%     std_train_F2 = std(F2_log_p_train);    
%     F2_log_p_test = (F2_log_p_test-mu_train_F2)/std_train_F2;
    
    % Plot
    log_plot(F2_log_p_test, Threshold_F2, title_plot{1,i},Seizure_time)
end 

%% Optimal threshold 

% To find the optimal threshold we define objective function as the
% distance from the ROC curve to the point (0,1) that is 
%           C(t) = sqrt(1 - TPR(t)^2  + FPR(t)^2)

clear FPR_matrix TPR_matrix
var_target_F1 = 90;
var_target_F2 = 77;
for i = 1:length(Feature1_cell_test)
     % Get seizure
    seizure_before = seizure{1,i};
    
    % Adjust seizure if we take window not equal 1 sek
    seiz_start = floor(seizure_before(1)/num_sec);
    seiz_end = ceil(seizure_before(end)/num_sec);
    seizure_new = seiz_start:seiz_end;
    
%     Get info for feature 1
    Data_train_F1 = Y_train_F1{1,i};
    Data_test_F1 = Y_test_F1{1,i};
    Data_F1 = [Data_train_F1 Data_test_F1];
    size_train = length(Data_train_F1)/(length(Data_train_F1) + length(Data_test_F1)); 
    
    [FPR, TPR, AreaROC,num_prin_vec_F1, f] = ROC_PC_spectral(Data_F1, var_explained_F1{1,i}, length_window,size_train,var_target_F1, seizure_new, title_plot{1,i}, plot_name{1,i});

    C1 = sqrt((1 - TPR).^2 + FPR.^2);
    C1_min = min(C1);
    C1_min_index = find(C1 == C1_min);
    
    C_ratio(i) = C1_min_index/length(FPR);
    optimal_threshold_F1(i) = f(C1_min_index);
    
    Sensitivity_F1(i) = TPR(C1_min_index);
    Specificity_F1(i) = 1 - FPR(C1_min_index);
    
    h2 = figure()
    plot(FPR,TPR)
    hold on 
    opt = plot(FPR(C1_min_index), TPR(C1_min_index), '*')
    plot(FPR,FPR, '--')
    xlabel('FPR')
    ylabel('TPR')
    legend([opt], {'Optimal threshold'}, 'Location', 'southeast')
    title(['Feature set 1 - ' title_plot{1,i}])
%     saveas(h2, sprintf('ROC_OptThres_F1_%s', plot_name{1,i}),'epsc') 
    
    
    % Get info for feature 2
    Data_train_F2 = Y_train_F2{1,i};
    Data_test_F2 = Y_test_F2{1,i};
    Data_F2 = [Data_train_F2 Data_test_F2];
    size_train = length(Data_train_F2)/(length(Data_train_F2) + length(Data_test_F2));     
    
    [FPR, TPR, AreaROC,num_prin_vec_F2, f] = ROC_PC_spectral(Data_F2, var_explained_F2{1,i}, length_window,size_train,var_target_F2, seizure_new, title_plot{1,i}, plot_name{1,i});
    
    C1 = sqrt((1 - TPR).^2 + FPR.^2);
    C1_min = min(C1);
    C1_min_index = find(C1 == C1_min);
    
    C_ratio(i) = C1_min_index/length(FPR);
    optimal_threshold_F2(i) = f(C1_min_index);
    
    Sensitivity_F2(i) = TPR(C1_min_index);
    Specificity_F2(i) = 1 - FPR(C1_min_index);
    
    h1 = figure()
    plot(FPR,TPR)
    hold on 
    opt = plot(FPR(C1_min_index), TPR(C1_min_index), '*')
    plot(FPR,FPR, '--')
    xlabel('FPR')
    ylabel('TPR')
    legend([opt], {'Optimal threshold'}, 'Location', 'southeast')
    title(['Feature set 2 - ' title_plot{1,i}])
%     saveas(h1, sprintf('ROC_OptThres_F2_%s', plot_name{1,i}),'epsc')    
end

   
Sensitivity_F1 = Sensitivity_F1';
Specificity_F1 = Specificity_F1';
Sensitivity_F2 = Sensitivity_F2';
Specificity_F2 = Specificity_F2';



%%
figure()
plot(optimal_threshold_F1)
hold on 
plot(optimal_threshold_F2)
legend('F1','F2')

max(optimal_threshold_F1)

max(optimal_threshold_F2)

Threshold_F1 = median(optimal_threshold_F1)

Threshold_F2 = median(optimal_threshold_F2)

optimal_threshold_F1 = optimal_threshold_F1'
optimal_threshold_F2 = optimal_threshold_F2'

mean(optimal_threshold_F1)
mean(optimal_threshold_F2)

std(optimal_threshold_F1)
std(optimal_threshold_F2)
