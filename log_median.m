function [log_p, log_p_train, log_p_test, ten_worst] = log_median(E_train, E, varargin)
% Log likelihood
% This function calculates the log-likelihood for simple gaussian
% distribution. 

% Input 
%       E_train: Training data
%       E: The whole data set

% Optional
%       Plot: 
%           Plot_boleen = true/false
%           Plot_title =  'Title'
%           Fraction worst
%      Seizure:
%           Seizure_boleen = true/false
%           Seizure_time = [start:end] (sek)

% Output
%       log_p: log likelihood for all data
%       log_p_train: log liklehood for train data 
%       log_p_test: log liklehood for test data 

% Get varargin
% only want 5 optional inputs at most
numvarargs = length(varargin);
% if numvarargs > 8
%     error('log_normal requires at most 7 optional inputs');
% end
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'Plot' false 'Missing title' 'Missing_name' 0.05 'Seizure' false [] 10};
optargs(1:numvarargs) = varargin;
[Plot, Plot_boleen, Plot_title  plot_name frac_worst Seizure Seizure_boleen Seizure_time bin_size] = optargs{:};

% We want to adjust the Gaussian distribution model to be more robust to
% outlier. Therefore we will look at the median instead of mu and replace
% Sigma for a matrix with MAD values on the diagonal. 
[train_row train_window] = size(E_train);

% E_train_org = E_train;
% E_train = E_train - mean(E_train(:));
% E_train = E_train/std(E_train(:));
% 
% E = E - mean(E_train_org(:));
% E = E/std(E_train_org(:));


M = median(E_train,2);
MAD = median(abs((E_train - M)),2);
MAD_matrix = diag(MAD);


num_window = length(E);
log_p = zeros(1,num_window);
for i = 1:num_window
%     log_p(1,i) = -0.5*log(norm(Sigma)) - 0.5*((E(i,:)'-mu)'*inv(Sigma)*(E(i,:)'-mu));
    log_p(1,i) = -0.5*log(det(MAD_matrix)) - 0.5*((E(:,i)-M)'*(MAD_matrix\(E(:,i)-M)));
end 

log_p_train = log_p(1:train_window);
log_p_test = log_p(train_window+1:end);
log_p_sort = sort(log_p_test); 
if frac_worst == 0 || floor(frac_worst*length(log_p_sort)) == 0
    ten_worst = 0;
else
    if floor(frac_worst*num_window) == 0
        ten_worst = zeros(1,num_window);
    else
    
    ten_pro = log_p_sort(floor(frac_worst*length(log_p_sort)));

    [I,J] = find(ten_pro > log_p_test);

    ten_worst = zeros(1, length(log_p_test));
    ten_worst(log_p_test <= ten_pro) = 1;
    ten_worst(log_p_test > ten_pro) = 0;
    
    log_worst = zeros(1,length(log_p_test));
    log_worst(log_p_test <= ten_pro) = log_p_test(log_p_test <= ten_pro); 

    end
end
if Plot_boleen    
    figure()
    subplot(2,1,1)
    if Seizure_boleen
        bar(Seizure_time(1):Seizure_time(2), ten_worst(Seizure_time(1):Seizure_time(2)), 'r','EdgeColor', 'r')
        hold on 
        bar(1:Seizure_time(1)-1, ten_worst(1:Seizure_time(1)-1), 'b','EdgeColor', 'b')
        hold on
        bar((Seizure_time(2)+1):num_window, ten_worst(Seizure_time(2)+1:end), 'b','EdgeColor', 'b')
        legend('Seizure', 'Non-Seizure')
    else
        bar(ten_worst, 'b')
    end
    title(['Most abnormal datapoints(' num2str(frac_worst*100) '%), or outliers, for ' Plot_title])
    axis([0 num_window 0 1.2])
    xlabel('Time (s)')
    yticks([1])
    yticklabels('Outliers')
    subplot(2,1,2)
    histogram(J,floor(num_window/bin_size), 'FaceColor', 'k', 'EdgeColor', 'k')
    axis([0 num_window 0 bin_size])
    title(['Outliers histogram, where each bin size is ' num2str(bin_size) 'seconds'])  
    xlabel('Time (s)')
    ylabel('Numbers of outliers')  
end


end 
