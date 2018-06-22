%% Load data 
load('EEG_clean_54.mat')
EEG_54 = EEG_clean_54;
Data_54 = EEG_54.data;

load('EEG_clean_55.mat')
EEG_55 = EEG_clean_55;
Data_55 = EEG_55.data;

% load('EEG_clean_56.mat')
% EEG_56 = EEG_clean_56;
% Data_56 = EEG_56.data;

load('EEG_clean_65.mat')
EEG_65 = EEG_clean_65;
Data_65 = EEG_65.data;

% load('EEG_clean_68.mat')
% EEG_68 = EEG_clean_68;
% Data_68 = EEG_68.data;

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


% load('EEG_clean_95_dont_use.mat')
% EEG_95 = EEG_clean_95_dont_use;
% Data_95 = EEG_95.data;

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
Data_pre = {Data_54, Data_55, Data_65, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};

%Data2 = {Data_54, Data_55, Data_56, Data_65, Data_68, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, EEG_clean_95_dont_use, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
Data_IED_pre = {Data_54,Data_65,Data_70,Data_83, Data_86, Data_90, Data_100};
Data_non_IED_pre = {Data_55, Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};

% Not using 87 and 104
%Data_non_IED_pre = {Data_55, Data_69, Data_82, Data_85, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_105};

%Data_non_IED_pre = {Data_55,  Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};
%Data_non_IED_pre = {Data_97,  Data_103, Data_104, Data_55,  Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_98, Data_99, Data_101,  Data_105};
%Data_non_IED_length = {length(Data_97),  Data_103, Data_55,  Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_98, Data_99, Data_101, Data_104, Data_105};

title_plot = {'Patient 54', 'Patient 55', 'Patient 65','Patient 69', 'Patient 70', 'Patient 82', 'Patient 83', 'Patient 85', 'Patient 86', 'Patient 87','Patient 90', 'Patient 92', 'Patient 93', 'Patient 94', 'Patient 97', 'Patient 98', 'Patient 99', 'Patient 100', 'Patient 101', 'Patient 103', 'Patient 104', 'Patient 105'};
title_plot2 = {'Patient 54',  'Patient 65', 'Patient 70',  'Patient 83', 'Patient 86','Patient 90', 'Patient 100'};
title_plot3 = {'Patient 97', 'Patient 98', 'Patient 99', 'Patient 101', 'Patient 103', 'Patient 104', 'Patient 105'};
plot_name = {'Chb54', 'Chb55',  'Chb65', 'Chb69','Chb70', 'Chb82', 'Chb83', 'Chb85', 'Chb86', 'Chb87', 'Chb90', 'Chb92', 'Chb93', 'Chb94', 'Chb97', 'Chb98','Chb99', 'Chb100', 'Chb101', 'Chb103', 'Chb104', 'Chb105'};

%% Take out the first and last 60 seconds
clear_sec = 60;
sample_rate = 128;
num_points = clear_sec * sample_rate;

% Clean IED data 
for i = 1:length(Data_pre)
    data = Data_pre{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_pre{1,i} = data;
end 

% Clean IED data 
for i = 1:length(Data_IED_pre)
    data = Data_IED_pre{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_IED_pre{1,i} = data;
end 

% Clean NON IED data 
for i = 1:length(Data_non_IED_pre)
    data = Data_non_IED_pre{1,i};
    data = data(:,num_points+1:end-num_points);
    
    Data_non_IED_pre{1,i} = data;
end

%% not using MAD

%Data = {Data_54, Data_55, Data_65, Data_68, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
%Data2 = {Data_54, Data_55, Data_56, Data_65, Data_68, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, EEG_clean_95_dont_use, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
%Data_IED = {Data_54,Data_65,Data_70,Data_83,Data_86, Data_90, Data_100};
%Data_non_IED = {Data_55,  Data_68, Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};
Data = {Data_54, Data_55, Data_65, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
Data_IED = {Data_54,Data_65,Data_70,Data_83, Data_86, Data_90, Data_100};
%Data_non_IED_pre = {Data_55, Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};

% Not using 87 and 104
Data_non_IED = {Data_55, Data_69, Data_82, Data_85, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_105};
%% Find variance in the data before MAD

% All data
data_all = [];
for i = 1:length(Data_pre) 
    
    data_all = [data_all var(Data_pre{1,i}')'];
end
variance_all = data_all;


% All data non
data_all_non = [];
for i = 1:length(Data_non_IED_pre) 
    data_all_non = [data_all_non var(Data_non_IED_pre{1,i}')'];
end
variance_all_non = data_all_non;


% All data with
data_all_with = [];
for i = 1:length(Data_IED_pre) 
    data_all_with = [data_all_with var(Data_IED_pre{1,i}')'];
end
variance_all_with = data_all_with;

%% Devide all data in every channel by the MAD of every channel
Data = {};
Data_IED = {};
Data_non_IED = {};

% All the data
for i = 1:length(Data_pre)
    data = Data_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
%     std_all = std(data')';    
    data_new = data./MAD;
%     data_new = data./std_all;
    Data{1,i} = data_new;
end

% Data with IED
for i = 1:length(Data_IED_pre)
    data = Data_IED_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
%     std_all = std(data')';     
    data_new = data./MAD;
%     data_new = data./std_all;
    Data_IED{1,i} = data_new;
end

% Data without IED
for i = 1:length(Data_non_IED_pre)
    data = Data_non_IED_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
        
%     std_all = std(data')';     
    data_new = data./MAD;
%     data_new = data./std_all;
    Data_non_IED{1,i} = data_new;
end
%% Find the variance in the data before and after MAD

% All data
data_all = [];
for i = 1:length(Data) 
    data_all = [data_all var(Data{1,i}')'];
end
variance_all_MAD = data_all;


% All data non
data_all_non = [];
for i = 1:length(Data_non_IED) 
    data_all_non = [data_all_non var(Data_non_IED{1,i}')'];
end
variance_all_non_MAD = data_all_non;


% All data with
data_all_with = [];
for i = 1:length(Data_IED) 
    data_all_with = [data_all_with var(Data_IED{1,i}')'];
end
variance_all_with_MAD = data_all_with;


%%
mean_var_all = mean(variance_all')';
mean_var_all_MAD = mean(variance_all_MAD')';

mean_var_all_non = mean(variance_all_non')';
mean_var_all_non_MAD = mean(variance_all_non_MAD')';

mean_var_all_with = mean(variance_all_with')';
mean_var_all_with_MAD = mean(variance_all_with_MAD')';

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
 Feature_cell = {};
 Feature_cell_F1 = {};
for k = 1:length(Data_IED)
    data = Data_IED{1,k};
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
    Feature1_cell_F1{1,k} = GetFeatures1(data, length_window);
    Feature_cell{1,k} = Feature_matrix;
    
end    

% GET DATA WITHOUT IED
  j = 1;
Feature_cell_train = {};
Feature_cell_train_F1 = {};
for k = 1:length(Data_non_IED)
    data = Data_non_IED{1,k};
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
    Feature_cell_train_F1{1,k}= GetFeatures1(data, length_window);
    Feature_cell_train{1,k} = Feature_matrix;
  
end 


%% split Feature cell train to train and test using only F1
Data_train = [];
Y_train = {};
Y_test_non = {};
Y_test_with = {};
for i = 1:length(Feature_cell_train) 
    Data_train = [Data_train Feature_cell_train_F1{1,i}];
end

Y_train{1,1} = Data_train;

for i = 7:length(Feature_cell_train_F1) 
   Y_test_non{1,i-6} =  Feature_cell_train_F1{1,i};
end

Y_test_with  = Feature1_cell_F1;

%%

Feature_non_IED_length = [];
for k = 1:length(Feature_cell_train)
    Feature_non_IED_length(k) = length(Feature_cell_train{1,k});
end
%% Split the data to train and test set

all_features = 84;
Data_train = [];
Feature_test_non = {};
Feature_test_with = Feature_cell;
%temp = [1 2  3 7 8 9 13  14  15 19 20 21 25  26 27 31  32 33  37 38 39  43 44 45  49 50  51 55 56  57 61 62 63 67  68 69 73  74  75 79 80 81 ];

% for i = 1:length(Feature_cell_train) 
%     Data_train = [Data_train Feature_cell_train{1,i}];
% end
for i = 1:length(Feature_cell) 
    Data_train = [Data_train Feature_cell{1,i}];
end

Data_test_non = [];
start_test = 9;
for i = start_test:length(Feature_cell_train)
    Data_test_non = [Data_test_non Feature_cell_train{1,i}];
    Feature_test_non{1,i-start_test+1} = Feature_cell_train{1,i};
end

Data_test_with = [];
for i = 1:length(Feature_cell)
    Data_test_with = [Data_test_with Feature_cell{1,i}];
    %Feature_test_with{1,i} = Feature_cell{1,i};
end

%% If we dont want to standardize
F_stand_train{1,1} = Data_train;
F_stand_test_non = Feature_test_non;
F_stand_test_with =Feature_test_with;

%% Standardize data 

F_stand_train = {};
F_stand_test_non = {};
F_stand_test_with = {};

% Find the mean and standard deviation of the training set
mu_train = mean(Data_train,2);
std_train = std(Data_train')';

% Standardize the training set
F_stand_train{1,1} = (Data_train - mu_train)./std_train;
% F_stand_train{1,1} = (Data_train - mu_train);

for k = 1:length(Feature_test_non)
    F_test_non = Feature_test_non{1,k};
    F_test_with = Feature_test_with{1,k};
    
%     F_stand_test_non{1,k} = (F_test_non - mu_train);
%     F_stand_test_with{1,k} = (F_test_with - mu_train);

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
    var_explained_non{1,k} = var_explained;
    
end

%% Visualize the features, only using the first 2 PCs


Data_train_stand = Y_train{1,1}(1:6,:);

Data_test_non_stand = [];
for i = 1:length(Y_test_non)
    Data_test_non_stand = [Data_test_non_stand Y_test_non{1,i}(1:2,:)];
end

Data_test_with_stand = [];
for i = 1:length(Y_test_with)
    Data_test_with_stand = [Data_test_with_stand Y_test_with{1,i}(1:2,:)];
end
% lenght_stand = [length(Data_test_with_stand),length(Data_test_non_stand),length(Data_train_stand)];
% min_length_stand = min(lenght_stand);
num_samples = 10000;
Data_train_rand = datasample(Data_train_stand',num_samples);
Data_non_rand = datasample(Data_test_non_stand',num_samples);
Data_with_rand = datasample(Data_test_with_stand',num_samples);

Data_all = [ Data_train_rand ;Data_non_rand; Data_with_rand]; 


% Scatterplot with histogram 
 test_train = {};
 for i = 1:num_samples*3
     if i < num_samples+1
         test_train{i,1} = 'Train';
     elseif i < num_samples*2+1 && i > num_samples
          test_train{i,1} = 'Non';
     else
         test_train{i,1} = 'With';
     end 
 end  
x = Data_all(:,1);
y = Data_all(:,2);

figure()
scatterhist(x,y,'Group',test_train,'Kernel','on')
title('Train, Test without and Test with IED')

%% Visualizing - For all data points

Data_all2 = [ Data_train_stand' ;Data_test_non_stand'; Data_test_with_stand'];
 test_train2 = {};
 for i = 1:(length(Data_train_stand)+length(Data_test_non_stand)+length(Data_test_with_stand))
     if i < length(Data_train_stand)+1
         test_train2{i,1} = 'Train';
     elseif i < length(Data_train_stand)+length(Data_test_non_stand)+1 && i > length(Data_train_stand)
          test_train2{i,1} = 'Non';
     else
         test_train2{i,1} = 'With';
     end 
 end  
x = Data_all2(:,1);
y = Data_all2(:,2);
figure()
scatterhist(x,y,'Group',test_train2,'Kernel','on')
title('Train, Test without and Test with IED')

%%
Data_train_stand = Y_train{1,1}(7:14,:);
index_vec = zeros(1,6);
index = 0;
for k = 1:6 %length(Feature_cell_train)
    index_vec(k) = index; 
    index = index + Feature_non_IED_length(k);
end 
%index_vec(k+1) = index;   
index_vec(1) = 1;


figure()
plot(Data_train_stand', '*')
hold on
plot(index_vec,Data_train_stand(:,index_vec) , 'r*')


%% Variance explained

var_target = 1:0.5:99.5;
for i = 1:length(var_target)
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target(i)
        num_prin = num_prin + 1;
    end 
    num_prin_vec(i) = num_prin;  
end 
num_prin_vec(i+1) = sum(var_explained>0);
var_target = [var_target 100];

figure()
plot(num_prin_vec,var_target)
xlabel('Number of PCs')
ylabel('% of variance explained')
title('Principal component analysis')

figure()
plot(cumsum(var_explained))
xlabel('Number of PCs')
ylabel('% of variance explained')
title('Principal component analysis')
%% Choosing 2, 5, 10 and 20 PCs - calculate Log normal
num_PCs = 18;
frac_worst = 0.01; 

for i =1:length(Y_test_non)
    SeizStart=0;
    SeizEnd = 1;
    % Get info for data without IDE  
    Data_train = Y_train{1,1};
    Data_train = Data_train(1:num_PCs,:);
    
    Data_test_non = Y_test_non{1,i};
    Data_test_non = Data_test_non(1:num_PCs,:);
    
    Data = [Data_train Data_test_non];        
    num_window = size(Data,2);
    [log_p, log_p_train, log_p_test] = log_normal_clean(Data_train, Data, 'Plot', false, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', false, [SeizStart SeizEnd], 10);
%      [log_p, log_p_train, log_p_test] = log_median(Data_train, Data, 'Plot', false, title_plot{1,i}, plot_name{1,i}, frac_worst, 'Seizure', false, [SeizStart SeizEnd], 10);
    log_p_non{1,i} = log_p;
    log_p_train_non{1,i} = log_p_train;
    log_p_test_non{1,i} = log_p_test;
    
    % Get info for data with IDE    
    Data_train = Y_train{1,1};
    Data_train = Data_train(1:num_PCs,:);
    
    Data_test_with = Y_test_with{1,i};
    Data_test_with = Data_test_with(1:num_PCs,:);
    
    Data = [Data_train Data_test_with];        
    num_window = size(Data,2);
   [log_p, log_p_train, log_p_test ] = log_normal_clean(Data_train, Data, 'Plot', false, title_plot2{1,i}, plot_name{1,i}, frac_worst, 'Seizure', false, [SeizStart SeizEnd], 10);
%      [log_p, log_p_train, log_p_test ] = log_median(Data_train, Data, 'Plot', false, title_plot2{1,i}, plot_name{1,i}, frac_worst, 'Seizure', false, [SeizStart SeizEnd], 10);

    log_p_with{1,i} = log_p;
    log_p_train_with{1,i} = log_p_train;
    log_p_test_with{1,i} = log_p_test;
        
end

%% Find the worst 10 percent in the training data
frac_worst = 0.01;
log_p_sort_train = sort(log_p_train);
ten_pro = log_p_sort_train(floor(frac_worst*length(log_p_sort_train)));
[I,J] = find(ten_pro > log_p_sort_train);

ten_worst = zeros(1, length(log_p_train));
ten_worst(log_p_train <= ten_pro) = 1;
ten_worst(log_p_train > ten_pro) = 0;

log_worst = zeros(1,length(log_p_train));
log_worst(log_p_train <= ten_pro) = log_p_train(log_p_train <= ten_pro);
[I,J] = find(ten_pro > log_worst);

Feature_non_IED_length = [];
for k = 1:length(Feature_cell)
    Feature_non_IED_length(k) = length(Feature_cell{1,k});

    %Feature_non_IED_length(k) = length(Feature_cell_train{1,k});
end
figure
plot(log_p_train)
hold on
%index_vec = zeros(1,length(Feature_cell_train));
index_vec = zeros(1,length(Feature_cell));
index = 0;
for k = 1:length(Feature_cell)
    index_vec(k) = index; 
    index = index + Feature_non_IED_length(k);
end 
index_vec(k+1) = index;   
index_vec(1) = 1;
plot(index_vec, log_p_train(index_vec), 'r*')
legend('Log likelihood','Start of new subject')
xlabel('Time')
ylabel('Log likelihood')
xlim([ -100 (index+100)])
ylim([-6000 100])
%title('Log likelihood - With IED subjects')

% Find the mean of the log likelihood for the training set
mean_log_p_train = zeros(1,length(Feature_cell_train));
median_log_p_train = zeros(1,length(Feature_cell_train));
num_outliers = zeros(1,length(Feature_cell_train));
thres_all = -20;
for k = 1:length(Feature_cell)
    mean_log_p_train(k) = (1/Feature_non_IED_length(k)) * sum(log_p_train((index_vec(k)+1):index_vec(k+1)));
    median_log_p_train(k) = median(log_p_train((index_vec(k)+1):index_vec(k+1)));
    num_outliers(k) = sum(log_p_train((index_vec(k)+1):index_vec(k+1)) < thres_all);
end
mean_log_p_train_trans = mean_log_p_train';
num_outliers_trans = num_outliers';

% Plot to show subject  87 and 104
figure()
plot(log_p_train, 'b')
hold on
plot((index_vec(5):index_vec(6)), log_p_train(index_vec(5):index_vec(6)), 'g')
hold on
plot((index_vec(14):index_vec(15)),log_p_train(index_vec(14):index_vec(15)), 'm')
legend('Max log likelihood - all Non subjects', 'Subject 87', 'Subject 104')

%% Plot first two PCs for all 'non' subjects and mark 87 and 104

Data_train_stand = Y_train{1,1}(1:2,:);
Data_train_87 = Y_train{1,1}(1:2,(index_vec(5):index_vec(6)));
Data_train_104 = Y_train{1,1}(1:2,(index_vec(14):index_vec(15)));

figure()
plot(Data_train_stand(1,:), Data_train_stand(2,:), 'b*')
hold on
plot(Data_train_87(1,:), Data_train_87(2,:), 'g*')
hold on
plot(Data_train_104(1,:), Data_train_104(2,:), 'm*')
hold on
legend('PCs for all Non IED', 'Subject 87', 'Subject 104')
title('Principal components for all Non subjects')
xlabel('Principal component 1')
ylabel('Principal component 2')

%%
% Figure of Q plot not standardized
jump = 0.05;
figure()
Q_train = {};
t_train = {};

Q_test_non = {};
t_test_non = {};

Q_test_with = {};
t_test_with = {};

for i = 1:length(log_p_train_with)
    % Feature set 1 
    F_log_p_train = log_p_train_non{1,i};
    F_log_p_test = log_p_test_non{1,i};
    
    [Q_train, Q_test,t_train, t_test] = QplotNew(F_log_p_train, F_log_p_test, jump);
    Q_train_non{1,i} = Q_train;
    Q_test_non{1,i} = Q_test;
    t_train_non{1,i} = t_train;
    t_test_non{1,i} = t_test;

    
    subplot(2,2,1)
    hold on 
    plot(t_train,Q_train)
    title('Train')
    xlabel('\theta')
    ylabel('Q(\theta)')
%     axis([-30 5 0 0.3])
    subplot(2,2,2)
    hold on 
    plot(t_test,Q_test)
    title('Test Non')
    xlabel('\theta')
    ylabel('Q(\theta)')
%     axis([-30 5 0 0.3])

    % Feature set 2
    F_log_p_train = log_p_train_with{1,i};
    F_log_p_test = log_p_test_with{1,i};

    [Q_train, Q_test,t_train, t_test] = QplotNew(F_log_p_train, F_log_p_test, jump);
    Q_train_with{1,i} = Q_train;
    Q_test_with{1,i} = Q_test;
    t_train_with{1,i} = t_train;
    t_test_with{1,i} = t_test;
    
    subplot(2,2,3)
    hold on 
    plot(t_train,Q_train)
    title('Train')
    xlabel('\theta')
    ylabel('Q(\theta)')
%     axis([-30 5 0 0.3])
    subplot(2,2,4)
    hold on
    plot(t_test,Q_test)
    title('Test With')
    xlabel('\theta')
    ylabel('Q(\theta)')  
%     axis([-30 5 0 0.3])
end 

%%
Q_opt_with = 0.005:0.005:0.03;
Q_opt_non = 0.005:0.005:0.03;

t_test_thres_with = zeros(length(Q_opt_with),length(Q_test_with));
t_test_thres_non = zeros(length(Q_opt_non),length(Q_test_non));
Q_test_with_thres = zeros(length(Q_opt_with),length(Q_test_with));
Q_test_non_thres = zeros(length(Q_opt_non),length(Q_test_non));
for j= 1:length(Q_opt_with)
    for i = 1:length(Q_train_non)

        Q_train_temp =  find(Q_train_non{1,i} > Q_opt_non(j));
        Q_train_thres = Q_train_temp(1);

        t_train_non_thres = t_train_non{1,i}(Q_train_thres);

        t_test_temp = find(t_test_non{1,i} > t_train_non_thres);
        t_test_thres = t_test_temp(1);
        t_test_thres_non(j,i) = t_test_non{1,i}(t_test_thres);

        Q_test_non_thres(j,i) = Q_test_non{1,i}(t_test_thres);
    end
    
    for i = 1:length(Q_train_with)

        Q_train_temp =  find(Q_train_with{1,i} > Q_opt_with(j));
        Q_train_thres = Q_train_temp(1);

        t_train_with_thres = t_train_with{1,i}(Q_train_thres);

        t_test_temp = find(t_test_with{1,i} > t_train_with_thres);
        t_test_thres = t_test_temp(1);
        t_test_thres_with(j,i) = t_test_with{1,i}(t_test_thres);

        Q_test_with_thres(j,i) = Q_test_with{1,i}(t_test_thres);
    end
    

end

%% finding the number of novelties
num_PCs = 14;
novelty_with =  zeros(length(Q_opt_with),length(Q_test_with));
novelty_non = zeros(length(Q_opt_non),length(Q_test_non));

  
for j = 1:length(Q_opt_with)
    for i =1:length(Y_test_non)
        SeizStart=0;
        SeizEnd = 1;
        % Get info for data without IDE  
        Data_train = Y_train{1,1};
        Data_train = Data_train(1:num_PCs,:);

        Data_test_non = Y_test_non{1,i};
        Data_test_non = Data_test_non(1:num_PCs,:);

        Data = [Data_train Data_test_non];        
        num_window = size(Data,2);
        [log_p, log_p_train, log_p_test, ten_worst] = log_normal_clean(Data_train, Data, 'Plot', false, title_plot3{1,i}, plot_name{1,i},  Q_test_non_thres(j,i), 'Seizure', false, [SeizStart SeizEnd], 10);
%         [log_p, log_p_train, log_p_test, ten_worst] = log_median(Data_train, Data, 'Plot', false, title_plot3{1,i}, plot_name{1,i},  Q_test_non_thres(j,i), 'Seizure', false, [SeizStart SeizEnd], 10);
        log_p_non{1,i} = log_p;
        log_p_train_non{1,i} = log_p_train;
        log_p_test_non{1,i} = log_p_test;
        
        % Finding the number of novelties
        novelty_non(j,i) = sum(ten_worst);

        % Get info for data with IDE    
        Data_train = Y_train{1,1};
        Data_train = Data_train(1:num_PCs,:);

        Data_test_with = Y_test_with{1,i};
        Data_test_with = Data_test_with(1:num_PCs,:);

        Data = [Data_train Data_test_with];        
        num_window = size(Data,2);
        [log_p, log_p_train, log_p_test, ten_worst] = log_normal_clean(Data_train, Data, 'Plot', false, title_plot2{1,i}, plot_name{1,i}, Q_test_with_thres(j,i), 'Seizure', false, [SeizStart SeizEnd], 10);
%         [log_p, log_p_train, log_p_test, ten_worst] = log_median(Data_train, Data, 'Plot', false, title_plot3{1,i}, plot_name{1,i},  Q_test_with_thres(j,i), 'Seizure', false, [SeizStart SeizEnd], 10);
        log_p_with{1,i} = log_p;
        log_p_train_with{1,i} = log_p_train;
        log_p_test_with{1,i} = log_p_test;
        
        % Finding the number of novelties
        novelty_with(j,i) = sum(ten_worst);
    end

end
%novelty_ratio_non = novelty_non./Feature_non_IED_length;

novelty_with_trans = novelty_with';
novelty_non_trans = novelty_non';

%% Summarizing the novelties
sum_novel_non = zeros(1,length(Q_opt_with));
sum_novel_with = zeros(1,length(Q_opt_with));
for k = 1:length(Q_opt_with)
   sum_novel_non(k) =  sum(novelty_non(k,:));
   sum_novel_with(k) = sum(novelty_with(k,:));
    
end

%% run pca on Feature set 2


% Featurset 2 
Train_data_F2 = Data_train;
[U_F2, Y_trans_F2, latent_F2, tsquared_F2, var_explained] = pca(Train_data_F2', 'Centered', false);
Y_train_F2 = Y_trans_F2'; 

% Test on 'non IED' data
Test_data_F2_non = Data_test_non ;
Y_test_F2_non = U_F2'*Test_data_F2_non;
var_explained_F2 = var_explained;

%  Test on 'with IED' data
Test_data_F2_with = Data_test_with ;
Y_test_F2_with = U_F2'*Test_data_F2_with;
var_explained_F2 = var_explained;



    
% Take the three top features and compare
variance_used = sum(var_explained_F2(1:3));

Y_train_used = Y_train(1:3,:);
Y_test_used_non = Y_test_non(1:3,:);
Y_test_used_with = Y_test_with(1:3,:);

%% Make plot of the three features 

N = 3;
    
figure()
  axis([0 10e4  0 10e4]) 
  ax = axis;
for i = 1:(N-1)
    for j = (i+1):N
        plotno = (j-1) + (N-1)*(i-1);
        subplot(N-1,N-1, plotno);
        plot(Y_train_used(i,:), Y_train_used(j,:), 'go',...
        Y_test_used_non(i,:), Y_test_used_non(j,:), 'bo',...
        Y_test_used_with(i,:), Y_test_used_with(j,:), 'ro')
        hold on 
        legend('Non IED data - Train','Non IED data - Test','IED data - Test')
      xlabel(sprintf('%d. Feature', i));
      ylabel(sprintf('%d. Feature', j))
      %axis(ax) 
    end
end

%%


