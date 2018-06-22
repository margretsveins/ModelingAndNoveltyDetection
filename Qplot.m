function [Q_train, Q_test] = Qplot(log_p, log_p_train, log_p_test, t)


total_train_points = length(log_p_train);
total_test_points = length(log_p_test);
Q_train = zeros(1,length(t));
Q_test = zeros(1,length(t));
for i = 1:length(t)
    num_outlier_train = sum(log_p_train<t(i));
    Q_train(i) = num_outlier_train/total_train_points;
    num_outlier_test = sum(log_p_test<t(i));
    Q_test(i) = num_outlier_test/total_test_points;
end

end
