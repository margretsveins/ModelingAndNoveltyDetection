% LOGPLOT for optimal value
e_max_opt = [0.2 0.4 0.4 0.2 0.2 0.2 0.2 0.3 0.1 0.3];
alpha_0_opt = [0.3 0.5 0.9 0.7 0.8 0.7 0.8 0.8 0.9 0.6];
tau_alpha_opt = [0.4 0.6 1 1 1.4 1.8 1 1.6 1.6 0.8];

% e_max_opt = [0.2 0.3	0.4	0.1	0.6	0.1	0.6	0.6	0.6	0.6];
% alpha_0_opt = [0.70	0.40	0.90	0.60	0.30	0.10	0.10	0.80	0.10	0.40];
% tau_alpha_opt = [ 1.2	1.0	0.2	0.8	1.2	0.2	1.8	0.2	0.6	1.0];
    
var_target = 77;

iter = 10;
FPR = [];
TPR = [];
for d = 1:length(Y_train_F2)  
    h = figure()
%     Get data     
    var_explained = var_explained_F2{1,d};
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 
    
    train = Y_train_F2{1,d}(1:num_prin,:);
    test = Y_test_F2{1,d}(1:num_prin,:);
    Seizure_time = seizure{1,d};
    
    % Standardise data
    train_mean = mean(train,2); 
    train_std = std(train')';
    train = (train-train_mean)./train_std;
    test = (test-train_mean)./train_std;   
%     
     e_max = e_max_opt(d);
     t_alpha = tau_alpha_opt(d);
     alpha_0 = alpha_0_opt(d);

    Q_t_max = 2*log(1./e_max);
    Outlier_matrix = [];
    d
    for q = 1:iter
        q
        [K_vec, y, sig2] = model2_new_cooling(train, 'e_max', e_max, 't_alpha', t_alpha, 'aplpha_0', alpha_0);
        outlier_index = [];
        for i = 1:length(test)
            x_t = test(:,i);
            Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
            Mahalanobis_dist = min(Mahalanobis_dist);
            if Mahalanobis_dist > Q_t_max
                outlier_index = [outlier_index i];
            end 
        end 
        ten_worst = zeros(1,length(test));
        ten_worst(outlier_index) = 1;
        Outlier_matrix(q,:) = ten_worst;     
        num_k(q) = K_vec(end);
    end    
    Outlier_cell3{1,d} = Outlier_matrix;
    num_k_cell2{1,d} = num_k;
     
%     ten_worst_mean = mean(Outlier_matrix);
%     ten_worst_mean(ten_worst_mean > 0) = 1;
% 
%     num_outlier = sum(ten_worst_mean>0);
%     num_seizure = sum(ten_worst_mean(Seizure_time)>0);
%     b1 = bar(ten_worst_mean, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
%     hold on
%     bar(Seizure_time, ten_worst_mean(Seizure_time), 'r','EdgeColor', 'r')        
%     legend('Non-Seizure','Seizure')
%     alpha(b1, 0.5)    
%     b1.EdgeAlpha = 0.10
%     title([title_plot{1,d} ' - ' num2str(num_outlier) ' outeliers, ' num2str(num_seizure) ' seizure points'])
%     axis([0 length(ten_worst_mean) 0 1.2])
%     xlabel('Time (s)')
%     set(gca, 'FontSize', 14)
% %     
%     TPR(d) = num_seizure/length(Seizure_time);
%     FPR(d) = (num_outlier - num_seizure)/(length(test) - length(Seizure_time));
%     mean_K(d) = mean(num_k);
%      saveas(h, sprintf('LogPlot_model2_subject_%s', plot_name{1,d}),'epsc')
    
end

%  Sensitivity = TPR'
%  Specificity = (1-FPR)'

%% 
for i = 1:10
    Seizure_time = seizure{1,i};
    Outlier_matrix = Outlier_cell{1,i};
%     Outlier_matrix = [Outlier_matrix; Outlier_cell{1,i}; Outlier_cell3{1,i} ];
    ten_worst_mean = sum(Outlier_matrix);
    ten_worst = ten_worst_mean > 2;

    num_outlier = sum(ten_worst>0);
    num_seizure = sum(ten_worst(Seizure_time)>0);

    TPR(i) = num_seizure/length(Seizure_time);
    FPR(i) = 1 - (num_outlier - num_seizure)/(length(test) - length(Seizure_time));

    h =figure()
    num_outlier = sum(ten_worst>0);
    num_seizure = sum(ten_worst(Seizure_time)>0);
    b1 = bar(ten_worst, 'b','EdgeColor', 'b', 'EdgeAlpha', 0.05)       
    hold on
    bar(Seizure_time, ten_worst(Seizure_time), 'r','EdgeColor', 'r')        
    legend('Non-Seizure','Seizure')
    alpha(b1, 0.5)    
    b1.EdgeAlpha = 0.10
    title(title_plot{1,i})
    axis([0 length(ten_worst) 0 1.2])
    xlabel('Time (s)')
    set(gca, 'FontSize', 14)
%     
    TPR(d) = num_seizure/length(Seizure_time);
    FPR(d) = (num_outlier - num_seizure)/(length(test) - length(Seizure_time));
    mean_K(d) = mean(num_k);
    saveas(h, sprintf('LogPlot_model2_subject_%s', plot_name{1,i}),'epsc')
    

end 

Sensitivity = TPR'
Specificity = FPR'

Outlier_matrix = Outlier_cell{1,7};
Seizure_time = seizure{1,7};
    ten_worst_mean = sum(Outlier_matrix);
    ten_worst = ten_worst_mean > 2;

    num_outlier = sum(ten_worst>0);
    num_seizure = sum(ten_worst(Seizure_time)>0);