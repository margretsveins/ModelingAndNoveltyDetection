%% Paths
% AT DTU

% Data
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\SampleData')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb02')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb04')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb05')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb07')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb10')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb10')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb13')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb14')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb16')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb20')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb21')
% addpath('C:\Users\s161286\Dropbox\Master thesis\Data\Model1_data\Chb22')

% % Code
% addpath('C:\Users\s161286\Dropbox\Master thesis\Code\Main')

% AT HOME
% Data
% 
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

% load('EEG_clean_56.mat')
% EEG_56 = EEG_clean_56;
% Data_56 = EEG_56.data;

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
Data_pre = {Data_54, Data_55, Data_65, Data_68, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
%Data2 = {Data_54, Data_55, Data_56, Data_65, Data_68, Data_69, Data_70, Data_82, Data_83, Data_85, Data_86, Data_87, Data_90, Data_92, Data_93, Data_94, EEG_clean_95_dont_use, Data_97, Data_98, Data_99, Data_100,Data_101,Data_103,Data_104,Data_105};
Data_IED_pre = {Data_54,Data_65,Data_70,Data_83,Data_86, Data_90, Data_100};
Data_non_IED_pre = {Data_55,  Data_68, Data_69, Data_82, Data_85, Data_87, Data_92, Data_93, Data_94, Data_97, Data_98, Data_99, Data_101, Data_103, Data_104, Data_105};
%Data_pre = {Data_02_pre, Data_04_pre, Data_05_pre, Data_07_pre, Data_10_pre, Data_13_pre, Data_14_pre, Data_16_pre, Data_20_pre, Data_21_pre,Data_22_pre};
%seizure = {seizure_02, seizure_04, seizure_05, seizure_07, seizure_10,seizure_13, seizure_14, seizure_16, seizure_20, seizure_21, seizure_22};
title_plot = {'Patient 54', 'Patient 55', 'Patient 65', 'Patient 68','Patient 69', 'Patient 70', 'Patient 82', 'Patient 83', 'Patient 85', 'Patient 86', 'Patient 87','Patient 90', 'Patient 92', 'Patient 93', 'Patient 94', 'Patient 97', 'Patient 98', 'Patient 99', 'Patient 100', 'Patient 101', 'Patient 103', 'Patient 104', 'Patient 105'};
title_plot2 = {'Patient 54',  'Patient 65', 'Patient 70',  'Patient 83', 'Patient 86','Patient 90', 'Patient 100'};

plot_name = {'Chb54', 'Chb55',  'Chb65', 'Chb68', 'Chb69','Chb70', 'Chb82', 'Chb83', 'Chb85', 'Chb86', 'Chb87', 'Chb90', 'Chb92', 'Chb93', 'Chb94', 'Chb97', 'Chb98','Chb99', 'Chb100', 'Chb101', 'Chb103', 'Chb104', 'Chb105'};

%% Devide all data in every channel by the MAD of every channel
Data = {};
Data_IED = {};
Data_non_IED = {};

for i = 1:length(Data_pre)
    data = Data_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
        
    data_new = data./MAD;
    Data{1,i} = data_new;
end
for i = 1:length(Data_IED_pre)
    data = Data_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
        
    data_new = data./MAD;
    Data_IED{1,i} = data_new;
end

for i = 1:length(Data_non_IED_pre)
    data = Data_non_IED_pre{1,i};
    
    M = median(data,2);
    MAD = median(abs((data - M)),2);
        
    data_new = data./MAD;
    Data_non_IED{1,i} = data_new;
end


%% Make the features for spectral
nwin = [];
nfft = nwin;
num_sec = 1;
sample_freq = 128;
length_window = sample_freq*num_sec;
noverlap = nwin * 0.5;
num_chan = 14;

% GET TEST DATA
 j = 1;
 Feature_cell = {};
 %Feature_norm_cell = {};
 %PSE_cell = {};
for k = 1:length(Data_IED)
    data = Data_IED{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    %Feature_matrix_norm = zeros(5*num_chan,num_window);
    %PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
        %Feature_matrix_norm(:,i) = E_normal;
        %PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature_cell{1,k} = Feature_matrix;
    %Feature_norm_cell{1,k} = Feature_matrix_norm;
    %PSE_cell{1,k} = PSE_matrix; 
    
end    

% GET CLEAN DATA
  j = 1;
 Feature_cell_train = {};
 %Feature_norm_cell_train = {};
 %PSE_cell_train = {};
for k = 1:length(Data_non_IED)
    data = Data_non_IED{1,k};
    num_window = floor(size(data,2)/length_window);
    Feature_matrix = zeros(6*num_chan,num_window);
    %Feature_matrix_norm = zeros(5*num_chan,num_window);
    %PSE_matrix = zeros(num_chan,num_window);
    j = 1;
    for i = 1:num_window
        % Get "window" of the data to summerize
        eeg_data = data(:,j:j+length_window-1)';  
        % Get the features
        [E, E_normal, PSE] = getFeatures(eeg_data, nwin, sample_freq, length_window, num_chan);
        Feature_matrix(:,i) = E;
%         Feature_matrix_norm(:,i) = E_normal;
%         PSE_matrix(:,i) = PSE;
        j = j+length_window;
    end
    
    Feature_cell_train{1,k} = Feature_matrix;
%     Feature_norm_cell_train{1,k} = Feature_matrix_norm;
%     PSE_cell_train{1,k} = PSE_matrix;    
end 


%%
format short
for b = 1:23
  time_length(b) = ceil((length(Data{1,b}) /128));
  time_length_2(b) = floor((length(Data{1,b}) /128)/60);
end

%% Check if the energy of the signal varies from patient to patient - using mean and median

% Data not with IED: Feature_cell_train
% Data with IED: Feature_cell
Data_non_mean = zeros(14,length(Feature_cell_train));
for i = 1:length(Feature_cell_train)
    
    Data_non_mean(:,i) = mean(Feature_cell_train{1,i}(6:6:84,:),2);
    Data_non_median(:,i) = median(Feature_cell_train{1,i}(6:6:84,:),2);

end


for i = 1:14
    Z = 1.960;
    n_train = length(Data_non_mean(i,:));
    std_train = std(Data_non_mean(i,:)');
    conf_int_train(i) = Z * std_train/sqrt(n_train);
end

Data_with_mean = zeros(14,length(Feature_cell));
for i = 1:length(Feature_cell)
    
    Data_with_mean(:,i) = mean(Feature_cell{1,i}(6:6:84,:),2);
    Data_with_median(:,i) = median(Feature_cell{1,i}(6:6:84,:),2);
end

for i = 1:14
    n_test = length(Data_with_mean(i,:));
    std_test = std(Data_with_mean(i,:)');
    conf_int_test(i) = Z * std_test/sqrt(n_test);
end 
figure()
subplot(2,1,1)
plot(mean(Data_non_mean,2), 'r*')
hold on
plot(mean(Data_with_mean,2), 'b*')
hold on
xlabel('channels')
ylabel('Mean energy of the signal')
title('Energy of the signal - mean of mean')
legend('No IED', 'With IED')

subplot(2,1,2)
plot(median(Data_non_median,2), 'r*')
hold on
plot(median(Data_with_median,2), 'b*')
hold on
xlabel('channels')
ylabel('Median energy of the signal')
title('Energy of the signal - median of median')
legend('No IED', 'With IED')

figure()
subplot(2,1,1)
plot(Data_non_mean, 'r*')
hold on
plot(Data_with_mean, 'b*')
hold on
xlabel('channels')
ylabel('Mean energy of the signal')
title('Energy of the signal - mean each channel')
legend('No IED', 'With IED')

subplot(2,1,2)
plot(Data_non_median, 'r*')
hold on
plot(Data_with_median, 'b*')
hold on
xlabel('channels')
ylabel('Median energy of the signal')
title('Energy of the signal - median of each channel')
legend('No IED', 'With IED')


%% All mean with confidence interval

xtick = 1:14;

figure()
hold on 
errorbar(mean(Data_non_mean,2), 2*conf_int_train, 'b')
%plot(error_train)
hold on 
%plot(error_test)
errorbar(mean(Data_with_mean,2), 2*conf_int_test, 'r')
legend('Non IED data', 'IED data')
xticks(xtick)
title('Energy of the signal - mean of each channel - 95% CI')
ylabel('Mean energy of the signal')
xlabel('Channels')





%% Make leave-one-out estimate for median and mean

diff_mean = zeros(length(Feature_cell), 14);
diff_median = zeros(length(Feature_cell), 14);
select = [1:length(Feature_cell),1:length(Feature_cell)];
for i = 1:length(Feature_cell)
    diff = mean(Data_with_mean(select(i+1:i+length(Feature_cell)-1)),2)- mean(Data_non_mean,2);
    diff_mean(i,:) = diff';
    
    diff2 = mean(Data_with_median(select(i+1:i+length(Feature_cell)-1)),2)- median(Data_non_median,2);
    diff_median(i,:) = diff2';
end

figure()
plot(diff_mean')
hold on
plot(zeros(1,14))
hold on 
title('Leave one out - mean')

figure()
plot(diff_median')
hold on
plot(zeros(1,14))
hold on 
title('Leave one out - median')

%% Get all the Energy of the signal
 Data_non_all = {};
 Data_with = {};


for i = 1:length(Feature_cell_train)
    Data_non_all{1,i} = Feature_cell_train{1,i}(6:6:84,:);
end

for i = 1:length(Feature_cell)
    
    Data_with_all{1,i} = Feature_cell{1,i}(6:6:84,:);
    
end

%% Standardize Energy of the signal for each patient
 
Data_non_stand = {};
Data_non_mean = [];
Data_non_std = [];
for i = 1:length(Feature_cell_train)
    Data_non = [];
    Data_non = Feature_cell_train{1,i}(6:6:84,:);
    Data_non_mean(:,i) = mean(Data_non,2);
    Data_non_std(:,i) = std(Data_non')';
    Data_non_stand{1,i} = (Data_non-Data_non_mean(:,i))./Data_non_std(:,i);
end

 
Data_with_stand = {};
Data_with_mean = [];
Data_with_std = [];
for i = 1:length(Feature_cell)
    Data_with = [];
    Data_with = Feature_cell{1,i}(6:6:84,:);
    Data_with_mean(:,i) = mean(Data_with,2);
    Data_with_std(:,i) = std(Data_with')';
    Data_with_stand{1,i} = (Data_with-Data_with_mean(:,i))./Data_with_std(:,i);
end



%% Check if the energy of the signal varies when using standardized data

% Data not with IED: Feature_cell_train
% Data with IED: Feature_cell
Data_non_mean_stand = zeros(14,length(Feature_cell_train));
Data_non_median_stand = zeros(14,length(Feature_cell_train));
for i = 1:length(Feature_cell_train)
    
    Data_non_mean_stand(:,i) = mean(Data_non_stand{1,i},2);
    Data_non_median_stand(:,i) = median(Data_non_stand{1,i},2);
end

for i = 1:14
    Z = 1.960;
    n_train = length(Data_non_mean_stand(i,:));
    std_train = std(Data_non_mean_stand(i,:)');
    conf_int_train_stand(i) = Z * std_train/sqrt(n_train);
end


Data_with_mean_stand = zeros(14,length(Feature_cell));
Data_with_median_stand = zeros(14,length(Feature_cell));

for i = 1:length(Feature_cell)
    
    Data_with_mean_stand(:,i) = mean(Data_with_stand{1,i},2);
    Data_with_median_stand(:,i) = median(Data_with_stand{1,i},2);
end

for i = 1:14
    n_test = length(Data_with_mean_stand(i,:));
    std_test = std(Data_with_mean_stand(i,:)');
    conf_int_test_stand(i) = Z * std_test/sqrt(n_test);
end 
figure()
subplot(2,1,1)
plot(mean(Data_non_mean_stand,2), 'r*')
hold on
plot(mean(Data_with_mean_stand,2), 'b*')
hold on
xlabel('channels')
ylabel('Mean energy of the signal')
title('Energy of the signal - standardizes - mean of mean')
legend('No IED', 'With IED')

subplot(2,1,2)
plot(median(Data_non_median_stand,2), 'r*')
hold on
plot(median(Data_with_median_stand,2), 'b*')
hold on
xlabel('channels')
ylabel('Median energy of the signal')
title('Energy of the signal - standardizes - median of median')
legend('No IED', 'With IED')


%% All mean with confidence interval for standardized

xtick = 1:15;

figure()
hold on 
errorbar(mean(Data_non_mean_stand,2), 2*conf_int_train_stand, 'b')
%plot(error_train)
hold on 
%plot(error_test)
errorbar(mean(Data_with_mean_stand,2), 2*conf_int_test_stand, 'r')
legend('Non IED data', 'IED data')
xticks(xtick)
title('Energy of the signal- STANDARDIZED - mean of each channel - 95% CI')
ylabel('Mean energy of the signal')
xlabel('Channels')
%%
figure()
for j = 1:length(Feature_cell_train)
    plot(Data_non_mean(:,j), 'r*')
    hold on
    if j < length(Feature_cell)
        plot(Data_with_mean(:,j), 'b*')
        hold on
    end
end
legend('No IED', 'With IED')

% for j = 1:length(Feature_cell)
%     plot(Data_with_mean(:,j), 'b*')
%     hold on
% end

%%
 Data_non = {};
 Data_with = {};


for i = 1:length(Feature_cell_train)
    Data_non{1,i} = Feature_cell_train{1,i}(6:6:84,:);
end

for i = 1:length(Feature_cell)
    
    Data_with{1,i} = Feature_cell{1,i}(6:6:84,:);
    
end

figure()
for j = 1:length(Data_non)
    plot(Data_non{1,j},'Color', rand(1,3))
    hold on
    if j < length(Data_with)    
        plot(Data_with{1,j},'Color', rand(1,3))
        hold on
    end
    xlabel('channels')
    ylabel('signal strength')
    title('Energy of the signal')
    legend('No IED', 'With IED')

end

%% Look at all the feature sets for both IED and non IED patients

all_features = 84;
Data_non_features = [];
Data_non_features_mean = zeros(all_features,1);
for i = 1:length(Feature_cell_train)
    Data_non_features = [Data_non_features Feature_cell_train{1,i}];
end
Data_non_features_mean = mean(Data_non_features,2);

conf_int_train_features = zeros(all_features,1);
for i = 1:all_features
    Z = 1.960;
    n_train = length(Data_non_features(i,:));
    std_train = std(Data_non_features(i,:)');
    conf_int_train_features(i) = Z * std_train/sqrt(n_train);
end
% conf_int_train_features = conf_int_train_features';


Data_with_features = [];
Data_with_features_mean = zeros(all_features,1);
for i = 1:length(Feature_cell)
    
    Data_with_features = [Data_with_features Feature_cell{1,i}];
end
Data_with_features_mean = mean(Data_with_features,2);

conf_int_test_features = zeros(all_features,1);
for i = 1:all_features
    n_test = length(Data_with_features(i,:));
    std_test = std(Data_with_features(i,:)');
    conf_int_test_features(i) = Z * std_test/sqrt(n_test);
end 
% conf_int_test_features = conf_int_test_features';
%% Make plot for all features agains one another

% channel 1
figure()
subplot(1,5,1)
plot(Data_non_features(1,:), Data_non_features(2,:), 'g*')
hold on
plot(Data_with_features(1,:), Data_with_features(2,:), 'r*')
hold on 
legend('Non IED data','IED data')
xlabel('Feature 1')
ylabel('Feature 2')

subplot(1,5,2)
plot(Data_non_features(1,:), Data_non_features(3,:), 'g*')
hold on
plot(Data_with_features(1,:), Data_with_features(3,:), 'r*')
hold on 
legend('Non IED data','IED data')
xlabel('Feature 1')
ylabel('Feature 3')

subplot(1,5,3)
plot(Data_non_features(1,:), Data_non_features(4,:), 'g*')
hold on
plot(Data_with_features(1,:), Data_with_features(4,:), 'r*')
hold on 
legend('Non IED data','IED data')
xlabel('Feature 1')
ylabel('Feature 4')

subplot(1,5,4)
plot(Data_non_features(1,:), Data_non_features(5,:), 'g*')
hold on
plot(Data_with_features(1,:), Data_with_features(5,:), 'r*')
hold on 
legend('Non IED data','IED data')
xlabel('Feature 1')
ylabel('Feature 5')

subplot(1,5,5)
plot(Data_non_features(1,:), Data_non_features(6,:), 'g*')
hold on
plot(Data_with_features(1,:), Data_with_features(6,:), 'r*')
hold on 
legend('Non IED data','IED data')
xlabel('Feature 1')
ylabel('Feature 6')


N = 6;
for b = 0:13
    
figure()
  axis([0 10e4  0 10e4]) 
  ax = axis;
for i = 1:(N-1)
    for j = (i+1):N
        plotno = (j-1) + (N-1)*(i-1);
        subplot(N-1,N-1, plotno);
        plot(Data_non_features(i+6*b,:), Data_non_features(j+6*b,:), 'go',...
        Data_with_features(i+6*b,:), Data_with_features(j+6*b,:), 'ro')
        hold on 
        legend('Non IED data','IED data')
      xlabel(sprintf('%d. Feature', i));
      ylabel(sprintf('%d. Feature', j))
      %axis(ax) 
     title(sprintf('%d. Channel', b+1))
    end
end
end

%% Make a plot of all the features

figure()
subplot(2,3,1)
plot(Data_non_features_mean(1:6:84,:),Data_with_features_mean(1:6:84,:), 'ro')
hold on
title('1. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 3*10e3 0 3*10e3])

subplot(2,3,2)
plot(Data_non_features_mean(2:6:84,:),Data_with_features_mean(2:6:84,:), 'ro')
hold on
title('2. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 1.5*10e3 0 1.5*10e3])

subplot(2,3,3)
plot(Data_non_features_mean(3:6:84,:),Data_with_features_mean(3:6:84,:), 'ro')
hold on
title('3. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 1*10e3 0 1*10e3])

subplot(2,3,4)
plot(Data_non_features_mean(4:6:84,:),Data_with_features_mean(4:6:84,:), 'ro')
hold on
title('4. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 1800 0 1800])

subplot(2,3,5)
plot(Data_non_features_mean(5:6:84,:),Data_with_features_mean(5:6:84,:), 'ro')
hold on
title('5. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 500 0 500])

subplot(2,3,6)
plot(Data_non_features_mean(6:6:84,:),Data_with_features_mean(6:6:84,:), 'ro')
hold on
title('6. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 3*10e3 0 3*10e3])


%%

figure()
errorbar(Data_non_features_mean(1:6:84,:),Data_with_features_mean(1:6:84,:), conf_int_test_features(1:6:84,1), conf_int_test_features(1:6:84,1), conf_int_train_features(1:6:84,:),conf_int_train_features(1:6:84,1), 'o')
hold on
plot(0:3*10e3, 0:3*10e3, 'r-')
%plot(mean(Data_non_features_mean(1:6:84,:),2)-std(Data_non_features_mean(1:6:84,:)'), mean(Data_non_features_mean(1:6:84,:),2)+std(Data_non_features_mean(1:6:84,:)'))
title('1. feature')
xlabel('Non IED data')
ylabel('IED data')
axis([0 3*10e3 0 3*10e3])
