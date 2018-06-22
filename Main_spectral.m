% This main function is for spectral use
%%%%%%%%%%%%%%%%%%%%%%%%%%%% functions used: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log_spectral
% E_matrix_spectral2
% num_pc
% log_normal
% channel_spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the clean data
EEG = EEG_clean_16_seiz;
EEG_data = EEG.data;
srate = EEG.srate;

% set time limits
t1 = EEG.xmin;
t2 = EEG.xmax;

% use constant window size with FFT
cycles = 0;
tlimits = [t1 t2]*1000;

% set the number of frequencies computed for each window
num_freq = 32;
num_win = length(EEG_data)/(srate/8);

% use the function log spectral to compute the ERSP
[freqs, times, ersp_mat] = log_spectral(EEG_data, srate, tlimits, num_freq, num_win);

%% Save the ersp_mat

ersp_mat_16_freq_24 = ersp_mat;
times_16_24 = times;
freqs_16_24 = freqs; 

%% START THE ANALYSIS 
% first rerank the data
data = ersp_mat_16_32;

% find the original rank
rank_data = rank(data);
[U_uni, Y_uni, latent_uni, tsquared_uni, explained_uni] = pca(data', 'Centered', false);
Y_uni = Y_uni';

% We want to reduce the variables, we will select the PC that explain X% of
% the variance in the data.
num_prin = 1;
while sum(explained_uni(1:num_prin)) < 90
    num_prin = num_prin + 1;
end 

% Set all the PC coefficent to zero that we want to get rid off 
U_uni(:,num_prin+1:end) = 0;

% Reconstructed data 
Y_rec = U_uni'*data;
data_uni_rec = U_uni*Y_rec;
rank_data_rec = rank(data_uni_rec);

%% FIRST APPROACH USING E_matrix for all channels

data = data_uni_rec;
% Test the training size and the variance explained
size_train = 0.8;
length_window = 8;

[E, E_train, var_explained] = E_matrix_spectral2(data, size_train,length_window,'Plot', false ,[], 'Standardize', true);
var_target= 90;
[E_pc, E_train_pc] = num_pc(E, E_train, var_explained, var_target);

%% Use the log norm function to see the outcome
frac_worst = 0.03;
[log_p, log_p_train, log_p_test] = log_normal(E_train_pc, E_pc, 'Plot', true, 'Patient 16 POWER SPEC: Seizure', frac_worst, 'Seizure', true, [2290 2299], 5);


%% DIFFERENT APPROACH USING E_matrix for each channel

data = data_uni_rec;
size_train = 0.4;
var_target= 80;
[E_pc_mat, E_train_pc_mat] = channel_spectra(data, var_target, size_train);

%% Use the log norm function to see the outcome
frac_worst = 0.03;
[log_p, log_p_train, log_p_test] = log_normal(E_train_pc_mat, E_pc_mat, 'Plot', true, 'Patient 16 POWER SPEC: Seizure', frac_worst, 'Seizure', true, [2290 2299], 5);





