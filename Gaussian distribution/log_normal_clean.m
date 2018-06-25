function [log_p, log_p_train, log_p_test, ten_worst] = log_normal_clean(E_train, E, varargin)
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
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'Plot' false 'Missing title' 'Missing_name' 0.05 'Seizure' false [] 10};
optargs(1:numvarargs) = varargin;
[Plot, Plot_boleen, Plot_title plot_name frac_worst Seizure Seizure_boleen Seizure_time bin_size] = optargs{:};

% Start by computing the mean vector and covarinace for the training data (E_train) 
[train_row train_window] = size(E_train);

mu = mean(E_train');
Sigma = cov(E_train');
error_term(1:length(mu)) = 10e-15;
% Sigma = Sigma  + diag(error_term);
num_window = length(E);
log_p = zeros(1,num_window);
for i = 1:num_window
%     log_p(1,i) = -0.5*log(norm(Sigma)) - 0.5*((E(:,i)-mu')'*inv(Sigma)*(E(:,i)-mu'));
    log_p(1,i) = -0.5*log(det(Sigma)) - 0.5*((E(:,i)-mu')'*(Sigma\(E(:,i)-mu')));
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
    h = figure()
    %subplot(2,1,1)
    if Seizure_boleen
        b1 = bar(ten_worst, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
        hold on
        bar(Seizure_time(1):Seizure_time(2), ten_worst(Seizure_time(1):Seizure_time(2)), 'r','EdgeColor', 'r')        
        legend('Non-Seizure','Seizure')
        alpha(b1, 0.5)    
        b1.EdgeAlpha = 0.10

    else
        b3 = bar(ten_worst, 'b', 'EdgeAlpha', 0.05)
        alpha(b3, 0.5)
        b3.EdgeAlpha = 0.10
    end
    hold on 
    title(['Most abnormal datapoints(' num2str(frac_worst*100) '%), or novelites, for ' Plot_title])
    axis([0 length(ten_worst) 0 1.2])
    xlabel('Time (s)')
    yticks([1])
    yticklabels('Novelties')
%     subplot(2,1,2)
%      histogram(J,floor(length(ten_worst)/bin_size), 'FaceColor', 'k', 'EdgeColor', 'k')
%     axis([0 length(ten_worst) 0 bin_size])
%     title(['Outliers histogram, where each bin size is ' num2str(bin_size) 'seconds'])  
%     xlabel('Time (s)')
%     ylabel('Numbers of outliers')  
    
     saveas(h, sprintf('ZeroOneSpectral_CleanTrain_%s', plot_name),'epsc')
end


end 
