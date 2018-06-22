function [error_train_mean_vector, error_test_mean_vector] = model2_optimalK(E, E_train, max_K, K_iter, nits, method)

K_vector = 2:max_K;
error_train_mean_iter = zeros(1, K_iter);
error_test_mean_iter = zeros(1, K_iter);
error_train_median_iter = zeros(1, K_iter);
error_test_median_iter = zeros(1, K_iter);
error_train_norm_iter = zeros(1, K_iter);
error_test_norm_iter= zeros(1, K_iter);
        
error_train_mean_vector = zeros(1, max_K-1);
error_test_mean_vector = zeros(1, max_K-1);
error_train_median_vector = zeros(1, max_K-1);
error_test_median_vector = zeros(1, max_K-1);
error_train_norm_vector = zeros(1, max_K-1);
error_test_norm_vector = zeros(1, max_K-1);

for k = K_vector
    for i = 1:K_iter
        [Etrain_arr, Etest_arr] = model2(E, E_train, k, nits, method, 'Plot', false);
        error_train_mean_iter(i) = mean(Etrain_arr);
        error_test_mean_iter(i) = mean(Etest_arr);
        error_train_median_iter(i) = median(Etrain_arr);
        error_test_median_iter(i) = median(Etest_arr);
        error_train_norm_iter(i) = -norm(Etrain_arr, 'inf');
        error_test_norm_iter(i)= -norm(Etest_arr, 'inf');
    end 
    error_train_mean_vector(k-1) = mean(error_train_mean_iter);
    error_test_mean_vector(k-1) = mean(error_test_mean_iter);
    error_train_median_vector(k-1) = mean(error_train_median_iter);
    error_test_median_vector(k-1) = mean(error_test_median_iter);
    error_train_norm_vector(k-1) = mean(error_train_norm_iter);
    error_test_norm_vector(k-1) = mean(error_test_norm_iter);
end 

figure()
plot(K_vector, error_train_mean_vector)
hold on 
plot(K_vector, error_test_mean_vector)
% hold on
% plot(K_vector, error_train_median_vector)
% hold on
% plot(K_vector, error_test_median_vector)
% hold on
% plot(K_vector, error_train_norm_vector)
% hold on
% plot(K_vector, error_test_norm_vector)
% legend('Train - mean', 'Test - mean', 'Train - median', 'Test - medain', 'Train - infinity norm', 'Test - infinity norm')
legend('Train - mean', 'Test - mean')
