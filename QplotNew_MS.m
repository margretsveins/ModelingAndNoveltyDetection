function [Q_train, Q_test,t_train, t_test] = QplotNew_MS(log_p_train, log_p_test, jump)

    t_train = sort(log_p_train);
    t_test = sort(log_p_test);
    
    t_train_cum = cumsum(abs(t_train));
    t_test_cum = cumsum(abs(t_test));
    total_train_points = length(log_p_train);
    total_test_points = length(log_p_test);

    Q_train = zeros(1,length(t_train));
    Q_test = zeros(1,length(t_test));
    
    for i = 1:length(t_train)
        num_outlier_train = sum(log_p_train<t_train(i));
        Q_train(i) = num_outlier_train/total_train_points;
    end
    
    for i = 1:length(t_test)
        num_outlier_test = sum(log_p_test<t_test(i));
        Q_test(i) = num_outlier_test/total_test_points;
    end

end