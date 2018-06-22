function [Dist_test, Dist_train, num_outlier_t_alpha, num_K] = Model2_alpha_t(DataTrain, alpha_0, tau_alpha, e_max, iter)

num_points = length(DataTrain);

% Number of partitions
k     = 5;

% Scatter row positions
pos   = randperm(num_points);

% Bin the positions into k partitions
edges = round(linspace(1,num_points+1,k+1));

% Partition Data
prtA  = cell(k,1);
for ii = 1:k
    idx      = edges(ii):edges(ii+1)-1;
    prtA{ii} = DataTrain(:,pos(idx)); % or apply code to the selection of A
end


Dist_test = zeros(length(tau_alpha), length(alpha_0));
Dist_train = zeros(length(tau_alpha), length(alpha_0));
num_K = zeros(length(tau_alpha), length(alpha_0));
num_outlier = zeros(length(tau_alpha), length(alpha_0));
Dist_test_iter = [];
Dist_train_iter = [];
Num_outlier_iter = [];
Num_K_iter = [];
Q_t_max =  2*log(1./e_max);
for j = 1:5
    j
    index_vec = 1:5;
    index_vec(j) = [];
    % Get test/train
    test = prtA{j};
    train = [];
    for k = index_vec
        train = [train prtA{k}];
    end
    
    Mahalanobis_dist_traint_vec = [];
    Mahalanobis_dist_test_vec = [];
    
    for a_0 = 1:length(alpha_0)      
        a_0       
        for t_a = 1:length(tau_alpha)
            if alpha_0(a_0) < tau_alpha(t_a)
                Outlier_matrix = [];
                for q = 1:iter
                    outlier_test = 0;
                    Outlier_index = [];

                    % Train model
                    [K_vec, y, sig2] = model2_new_cooling(train, 'e_max', e_max, 't_alpha', tau_alpha(t_a), 'aplpha_0', alpha_0(a_0));

                    % Calculate Mahalanobis dist for test
                    for i = 1:length(test)
                        x_t = test(:,i);
                        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
                        Mahalanobis_dist_test_vec(i) = min(Mahalanobis_dist);
                        if min(Mahalanobis_dist) > Q_t_max
                            outlier_test = outlier_test + 1;
                            Outlier_index =  [Outlier_index  i];
                        end 
                        ten_worst = zeros(1,length(test));
                        ten_worst(Outlier_index) = 1;
                        Outlier_matrix(q,:) = ten_worst; 
                    end 

                    % Calculate Mahalanobis dist for train
                    for i = 1:length(train)
                        x_t = train(:,i);
                        Mahalanobis_dist = diag((x_t - y')'*(x_t - y'))./sig2;   
                        Mahalanobis_dist_traint_vec(i) = min(Mahalanobis_dist);               
                    end 
                    Dist_test_iter(q) = mean(Mahalanobis_dist_test_vec);
                    Dist_train_iter(q) = mean(Mahalanobis_dist_traint_vec);
                    Num_K_iter(q) = K_vec(end);
                end
%                 num_outlier_t_alpha(t_a,a_0) = sum(mean(Outlier_matrix)>0); 
                num_outlier_t_alpha(t_a,a_0) = sum(Outlier_matrix>0);
                Dist_test_t_alpha(t_a,a_0) = mean(Dist_test_iter);
                Dist_train_t_alpha(t_a,a_0) = mean(Dist_train_iter);
                Num_k_t_alpha(t_a,a_0) =mean(Num_K_iter);  
            else
                num_outlier_t_alpha(t_a,a_0) = NaN;
                Dist_test_t_alpha(t_a,a_0) = NaN;
                Dist_train_t_alpha(t_a,a_0) = NaN;
                Num_k_t_alpha(t_a,a_0) = NaN;
            end
        end
%         num_outlier_matrix(:,a_0) = num_outlier_t_alpha;
%         Dist_test_matrix(:,a_0) = Dist_test_t_alpha;
%         Dist_train_matrix(:,a_0) = Dist_train_t_alpha;
%         Num_k_matrix(:,a_0) = Num_k_t_alpha;
    end
    Dist_test = Dist_test + Dist_test_t_alpha;
    Dist_train = Dist_train + Dist_train_t_alpha;
    num_outlier = num_outlier+ num_outlier_t_alpha;
    num_K = num_K+ Num_k_t_alpha;
end
    Dist_test = Dist_test./j;
    Dist_train = Dist_train./j;
    num_outlier = num_outlier./j;
    num_K = num_K+ Num_k_t_alpha./j;
end 