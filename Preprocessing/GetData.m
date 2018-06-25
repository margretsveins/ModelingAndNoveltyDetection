function EEG_uni = GetData(name)

% Import data 
EEG_bipolar = pop_biosig(name);

% Make sure that we use the right channels 
num_chanels = size(EEG_bipolar.data, 1);

% Import data, make sure that we take in the right channels
if num_chanels == 28
    % If we have 28 channels, we have to filter out the flat channels
    EEG_bipolar = pop_biosig(name,'channels',[1 2 3 4 6 7 8 9 11 12 14 15 16 17 19 20 21 22 24 25 26 27 28]);
    error_out = 0;
elseif num_chanels == 23
    error_out = 0;
else
    warning(['The number of channels are ' num2str(num_channels)])% <-- FUNCTION
    error_out = 1;
end

if error_out == 0
    EEG_uni = unipolar(EEG_bipolar);
    % ADD CHANNEL LOCATION
    % Load the channel loc file (homemade)
    channel_loc = load('channel_loc.mat');
    EEG_uni.chanlocs = channel_loc.EEG_uni.chanlocs;
    % Filter
    [EEG_uni, com, b] = pop_eegfiltnew(EEG_uni, [], 1, [], true, [], 0); % High pass filter for 0.1
elseif error_out == 1 
    EEG_uni.srate = 'error';
end 

end 