function [W, num_window] = Frames(Data, length_window) 
% Sampling rate: Have many samples carried per second, measured in Hz
%                Physiodata: 256 Hz

% We are going to create a window signal pieces where each window is 1 sek
% Thus, for each window we will create matrix for all the channels.
[num_var num_obs] = size(Data);
num_window = floor(num_obs/length_window);

clean_data = Data;

% Constructing Window matrix
size_cd = size(clean_data);
W = zeros(size_cd(1),length_window, floor(size_cd(2)/length_window)); % Window matrix

% Index for the clean data 
i_start = 1;

% Index for window matrix
j = 1;
while i_start + length_window - 1 <= num_obs
    % Create the window w
    w = clean_data(:,i_start:(i_start + length_window-1));
    % Store all the windos in W
    W(:,:,j) = w;
    
    i_start = i_start + length_window;
    j = j + 1;
end 