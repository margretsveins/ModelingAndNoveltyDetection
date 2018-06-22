function [error_train error_test error_train_norm error_test_norm] = Model_1_validation_RANDOM(Data, length_window,size_train,num_pc,varargin )
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
iter = 20;
for i = 1:length(size_train)
    for j = 1:iter
    [E, E_train, tf] = PCA_TrainTest_RANDOM(Data, size_train(i), length_window, 'PC', true, num_pc, false); % <-- FUNCTION
    [~, log_p_train, log_p_test] = log_normal(E_train, E, tf); % <-- FUNCTION
    error_train_iter(j) = abs(mean(log_p_train));
    error_test_iter(j) = abs(mean(log_p_test));
    error_train_norm_iter(j) = abs(median(log_p_train));
    error_test_norm_iter(j) = abs(median(log_p_test));
    end
    error_train(i) = (mean(error_train_iter));
    error_test(i) = (mean(error_test_iter));
    error_train_norm(i) = (median(error_train_norm_iter));
    error_test_norm(i) = (median(error_test_norm_iter));
end

if Plot_boleen
    Model_1_errorPlot(error_train, error_test, error_train_norm, error_test_norm,size_train, xAxis_jump)
end

end 