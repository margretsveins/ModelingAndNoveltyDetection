%%  Preprocessing SBS2 data
clear all
close all
clc
%%
addpath('/Users/margretsveinsdottir/Documents/MATLAB/eeglab14_1_1b')
addpath('/Users/margretsveinsdottir/Documents/DTU/Vor 2018/THESIS/Code/Lauras code/IC_MARC_current')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Data/First_run_pre')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Code/Margret')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Data')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Data/Seizure_data')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Data/First_run_pre')
addpath('/Users/margretsveinsdottir/Dropbox/Master thesis/Data/smartphone')

eeglab


%% DATA chosen from SBS2

run = {...
        '54_TRA.edf','55_DIA.edf','56_LY.edf','65_SYL.edf' ...    
        ,'68_MBA.edf','69_BAM.edf','70_SOW.edf','82_CAM.edf' ...
        ,'83_KER.edf','85_DIA.edf','86_CAM.edf','87_DIA (2).edf'...
        ,'90_DIA.edf','92_CON.edf','93_SAM.edf','94_DIA.edf'...
        ,'95_SID.edf','97_YOU.edf','98_KEB.edf','99_BAR.edf'...
        ,'100_TON.edf','101_MAM.edf','103_BAR.edf','104_DIA.edf'...
        ,'105_SOU.edf'...
       };
number_of_runs = length(run);
%%
% We will make a cell array to store all our EEG files.
EEG_all = cell(number_of_runs, 1);
EEG_process_all = cell(number_of_runs, 1);
EEG_clean_all = cell(number_of_runs, 1);
channel_diff = length(number_of_runs);

for i = 1:number_of_runs
    % Get unipolar data and use hight passfilters 
    EEG = pop_biosig(run{i});
    
    % Save channel locations
    channel_loc_SBS2 = load('channel_loc_SBS2.mat');
    EEG.chanlocs = channel_loc_SBS2.channel_loc_SBS2;
    EEG_all{i} = EEG;
    
    % Highpass filter
    [EEG_filter, com, b] = pop_eegfiltnew(EEG, [], 1, [], true, [], 0);
    EEG_process = clean_rawdata(EEG_filter, 5, -1, 0.85, 4, -1, -1);
    
    % Find the difference of channel after cleaning
    [n1,m1] = size(EEG.data);
    [n2,m2] = size(EEG_process.data);
    channel_diff(i) = n1-n2;
    
    % Interpolate
    EEG_int = pop_interp(EEG_process, EEG.chanlocs, 'spherical');
    EEG_process_all{i} = EEG_int;
    
end

%% Run ICA and remove artifacts

EEG_clean_all = cell(number_of_runs, 1);
ICA_diff = length(number_of_runs);

for i = 2:number_of_runs
    % Get unipolar data and use hight passfilters 
    EEG_process = EEG_process_all{i,1};
    
    % Run ICA and remove artifacts
    EEG_clean = cleanData(EEG_process); % <-- FUNCTION 
    EEG_clean_all{i} = EEG_clean;
    
    % Find the difference of channel after cleaning
    [n1,m1] = size(EEG.data);
    [n2,m2] = size(EEG_clean.icaweights);
    ICA_diff(i) = n1-n2;

end
%%
EEG_process_54 = EEG_process_all{1,1};
EEG_process_55 = EEG_process_all{2,1};
EEG_process_56 = EEG_process_all{3,1};
EEG_process_65 = EEG_process_all{4,1};
EEG_process_68 = EEG_process_all{5,1};
EEG_process_69 = EEG_process_all{6,1};
EEG_process_70 = EEG_process_all{7,1};
EEG_process_82 = EEG_process_all{8,1};
EEG_process_83 = EEG_process_all{9,1};
EEG_process_85 = EEG_process_all{10,1};
EEG_process_86 = EEG_process_all{11,1};
EEG_process_87 = EEG_process_all{12,1};
EEG_process_90 = EEG_process_all{13,1};
EEG_process_92 = EEG_process_all{14,1};
EEG_process_93 = EEG_process_all{15,1};
EEG_process_94 = EEG_process_all{16,1};
EEG_process_95 = EEG_process_all{17,1};
EEG_process_97 = EEG_process_all{18,1};
EEG_process_98 = EEG_process_all{19,1};
EEG_process_99 = EEG_process_all{20,1};
EEG_process_100 = EEG_process_all{21,1};
EEG_process_101 = EEG_process_all{22,1};
EEG_process_103 = EEG_process_all{23,1};
EEG_process_104 = EEG_process_all{24,1};
EEG_process_105 = EEG_process_all{25,1};

%%

EEG_pre_54 = EEG_all{1,1};
EEG_pre_55 = EEG_all{2,1};
EEG_pre_56 = EEG_all{3,1};
EEG_pre_65 = EEG_all{4,1};
EEG_pre_68 = EEG_all{5,1};
EEG_pre_69 = EEG_all{6,1};
EEG_pre_70 = EEG_all{7,1};
EEG_pre_82 = EEG_all{8,1};
EEG_pre_83 = EEG_all{9,1};
EEG_pre_85 = EEG_all{10,1};
EEG_pre_86 = EEG_all{11,1};
EEG_pre_87 = EEG_all{12,1};
EEG_pre_90 = EEG_all{13,1};
EEG_pre_92 = EEG_all{14,1};
EEG_pre_93 = EEG_all{15,1};
EEG_pre_94 = EEG_all{16,1};
EEG_pre_95 = EEG_all{17,1};
EEG_pre_97 = EEG_all{18,1};
EEG_pre_98 = EEG_all{19,1};
EEG_pre_99 = EEG_all{20,1};
EEG_pre_100 = EEG_all{21,1};
EEG_pre_101 = EEG_all{22,1};
EEG_pre_103 = EEG_all{23,1};
EEG_pre_104 = EEG_all{24,1};
EEG_pre_105 = EEG_all{25,1};

%%

EEG_clean_54 = EEG_clean_all{1,1};
EEG_clean_55 = EEG_clean_all{2,1};
EEG_clean_56 = EEG_clean_all{3,1};
EEG_clean_65 = EEG_clean_all{4,1};
EEG_clean_68 = EEG_clean_all{5,1};
EEG_clean_69 = EEG_clean_all{6,1};
EEG_clean_70 = EEG_clean_all{7,1};
EEG_clean_82 = EEG_clean_all{8,1};
EEG_clean_83 = EEG_clean_all{9,1};
EEG_clean_85 = EEG_clean_all{10,1};
EEG_clean_86 = EEG_clean_all{11,1};
EEG_clean_87 = EEG_clean_all{12,1};
EEG_clean_90 = EEG_clean_all{13,1};
EEG_clean_92 = EEG_clean_all{14,1};
EEG_clean_93 = EEG_clean_all{15,1};
EEG_clean_94 = EEG_clean_all{16,1};
EEG_clean_95 = EEG_clean_all{17,1};
EEG_clean_97 = EEG_clean_all{18,1};
EEG_clean_98 = EEG_clean_all{19,1};
EEG_clean_99 = EEG_clean_all{20,1};
EEG_clean_100 = EEG_clean_all{21,1};
EEG_clean_101 = EEG_clean_all{22,1};
EEG_clean_103 = EEG_clean_all{23,1};
EEG_clean_104 = EEG_clean_all{24,1};
EEG_clean_105 = EEG_clean_all{25,1};