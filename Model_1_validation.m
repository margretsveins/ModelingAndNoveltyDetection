function [error_train error_test error_train_norm error_test_norm] = Model_1_validation(Data, length_window,size_train,num_pc,varargin )
% This function calculates both the mean and norm of the log likelihood and
% plots up the error if that is wanted 

% Input:
%   Data
%   length_window
%   size_train
% Optional:
%   Plot
%   Plot_boleen
%   xAxis jump
% Output:
% error_train 
%   error_test 
%   error_train_norm 
%   error_test_norm


% Get varargin
% only want 5 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 7
    error('log_normal requires at most 7 optional inputs');
end
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'Plot' false 1};
optargs(1:numvarargs) = varargin;
[Plot, Plot_boleen, xAxis_jump] = optargs{:};

for i = 1:length(size_train)
    [E, E_train] = PCA_TrainTest(Data, size_train(i), length_window, 'PC', true, num_pc, false); % <-- FUNCTION
    [~, log_p_train, log_p_test] = log_normal(E_train, E); % <-- FUNCTION
%     [~, log_p_train, log_p_test] = log_median(E_train, E); % <-- FUNCTION
    error_train(i) = abs(mean(log_p_train));
    error_test(i) = abs(mean(log_p_test));
    error_train_norm(i) = abs(median(log_p_train));
    error_test_norm(i) = abs(median(log_p_test));
end

if Plot_boleen
    Model_1_errorPlot(error_train, error_test, error_train_norm, error_test_norm,size_train, xAxis_jump)
end

end 