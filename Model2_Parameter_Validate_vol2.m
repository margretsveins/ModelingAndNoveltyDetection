%% ALPHA_T
% Evalute paramters for GMM 

Dist_test_alphat_16to20 = {};
Dist_train_alphat_16to20 = {};
num_outlier_alphat_16to20 = {};
num_K_alphat_16to20 = {};
var_target = 77;
alpha_0 = 0.1:0.1:1;
tau_alpha = 0.2:0.2:2;
iter = 1;
e_max_opt = [0.2 0.3];
for d = 1:length(Data)
    d
    e_max = e_max_opt(d);
    var_explained = var_explained_F2{1,d};
    num_prin = 1;
    while sum(var_explained(1:num_prin)) < var_target
        num_prin = num_prin + 1;
    end 

    DataTrain = Y_train_F2{1,d}(1:num_prin,:);
    [Dist_test, Dist_train, num_outlier_test, num_K] = Model2_alpha_t(DataTrain, alpha_0, tau_alpha, e_max, iter);
  
    Dist_test_alphat_16to20{1,d} = Dist_test;
    Dist_train_alphat_16to20{1,d} = Dist_train;
    num_outlier_alphat_16to20{1,d} = num_outlier_test;
    num_K_alphat_16to20{1,d} = num_K;
    
%     dist_opt = Dist_test_e_max./sum(Dist_test_e_max); 
%     novelty_opt = mean_num_outlier./sum(mean_num_outlier);
% 
%     [value index] = min(dist_opt + novelty_opt);
%     e_max_opt(d) = e_max_vec(index);
%     
%     h = figure()
%     yyaxis left 
%     h1 = plot(Dist_test_e_max)
%     hold on 
%     h2 = plot(Dist_train_e_max)
%     ylabel('Mean Mahalanobis distance (D)', 'FontSize', 16)
%     hold on 
%     yyaxis right
%     plot(mean_num_outlier)
%     legend([h1 h2], {'Test', 'Train'})
%     xticks(1:length(e_max_vec))
%     xticklabels(e_max_vec)
%     ylabel('Mean number of outlier (nov)', 'FontSize', 16)
%     xlabel('\epsilon_{max}', 'FontSize', 16)
%     title(title_plot{1,d})
%     set(gca, 'FontSize', 14)
% % 
%     saveas(h, sprintf('E_max_subject_%s', plot_name{1,d}),'epsc')
end