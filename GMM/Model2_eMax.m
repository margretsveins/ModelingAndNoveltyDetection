function [Dist_test, Dist_train, num_outlier_test, num_K] = Model2_eMax(DataTrain, e_max_vec, iter)

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

% Define parameters
alpha_0 = 0.7;
t_alpha = 1;


Dist_test = [];
Dist_train = [];
num_K = [];
Dist_test_iter = [];
Dist_train_iter = [];
Num_outlier_iter = [];
Num_K_iter = [];

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
    
    for e = 1:length(e_max_vec)      
        e_max_vec(e)
        Q_t_max =  2*log(1./e_max_vec(e));
        Outlier_matrix = [];
        for q = 1:iter
            outlier_test = 0;
            Outlier_index = [];
            
            % Train model
            [K_vec, y, sig2] = model2_new_cooling(train, 'e_max', e_max_vec(e), 't_alpha', t_alpha, 'aplpha_0', alpha_0);

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
            Num_outlier_iter(q) = outlier_test;
            Num_K_iter(q) = K_vec(end);
        end
        Dist_test(j,e) = mean(Dist_test_iter);
        Dist_train(j,e) = mean(Dist_train_iter);
        num_outlier_test(j,e) = sum(Outlier_matrix>0);
        num_K(j,e) = mean(Num_K_iter);
    end
end

end 