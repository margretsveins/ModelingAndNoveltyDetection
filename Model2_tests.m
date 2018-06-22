% ONE PARAMETER
e_max = 0:0.05:0.8;
t_alpha = 1;
alpha_0= 0.7;

TPR_subject_cum = [];
FPR_subject_cum = [];
K_subject = [];
Outlier_subject = {};
var_target = 75; 
iter = 5;

for d = 1:length(Y_train_F2)
    
    % Get data     
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

    K_iter = [];
    TPR_cum = [];
    FPR_cum = [];
    for e = 1:length(e_max)
        d
        e_max(e)
        Q_t_max = 2*log(1./e_max(e));
        Outlier_matrix = [];
        for q = 1:iter
            q
            [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max(e), 't_alpha', t_alpha, 'aplpha_0', alpha_0);
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
            K_iter(q,e) = K_vec(end);        
        end 
        Outlier_subject{d,e} = Outlier_matrix;
        
        % Compute for TPR and FPR for "total outlier" 
        ten_worst = zeros(1,length(test));
        if q == 1             
            ten_worst(outlier_index) = 1;
        else
            ten_worst(mean(Outlier_matrix) > 0) = 1;
        end
        true_positive = sum(ten_worst(Seizure_time));
        False_postive = sum(ten_worst)-true_positive;
        num_seizure = length(Seizure_time);
        Condition_negative = length(test)-num_seizure;

        TPR = true_positive/num_seizure;
        FPR = False_postive/Condition_negative;        

        TPR_cum(1,e) = TPR;
        FPR_cum(1,e) = FPR;
    end
%     Outlier_subject{1,d} = Outlier_matrix;
    TPR_subject_cum(d,:) = TPR_cum;
    FPR_subject_cum(d,:) = FPR_cum;
    K_subject(:,:,d) = K_iter;       
end




%% TWO PARAMETERS
% Define parameters 

t_alpha = 0.2:0.2:2;
alpha_0= 0.1:0.1:1;
var_target = 77; 
iter = 5;


TPR_subject_cum = [];
FPR_subject_cum = [];
Outlier_subject_alpha_t = {};

% ------------------- LOOP OVER DATA -------------------------------------- 
for d = 1:length(Y_train_F2)
    % Get data     
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

    K_iter = [];
    TPR_cum = [];
    FPR_cum = [];
    
    e_max = e_opt(d);
    Q_t_max = 2*log(1./e_max);    
    
    % --------------------- LOOP FOR PARAMETER 1 --------------------------
    for a_0 = 1:length(alpha_0)
        % ------------------------ LOOP FOR PARAMETER 2 -------------------
        for t_a = 1:length(t_alpha)
            d
            alpha_0(a_0)
            t_alpha(t_a)
            if alpha_0(a_0) > t_alpha + 1
                TPR_cum(t_a,a_0) = 0;
                FPR_cum(t_a,a_0) = 1;
            else
                Outlier_matrix = [];
                % ---------------------- LOOP ITER ----------------------------
                for q = 1:iter
                    q
                    [K_vec, y, sig2] = model2_new_cooling(train,test, 'e_max', e_max, 't_alpha', t_alpha(t_a), 'aplpha_0', alpha_0(a_0));
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
                end 
                % -------------------------- END ITER ---------------------
                  % Compute for TPR and FPR for "total outlier" 
                ten_worst = zeros(1,length(test));
                if q == 1             
                    ten_worst(outlier_index) = 1;
                else
                    ten_worst(mean(Outlier_matrix) > 0) = 1;
                end
                true_positive = sum(ten_worst(Seizure_time));
                False_postive = sum(ten_worst)-true_positive;
                num_seizure = length(Seizure_time);
                Condition_negative = length(test)-num_seizure;

                TPR = true_positive/num_seizure;
                FPR = False_postive/Condition_negative;        

                TPR_cum(t_a,a_0) = TPR;
                FPR_cum(t_a,a_0) = FPR;
            end 
            % ------------------------- END IF ----------------------------   
        end
            % ------------------------- END PARAMETER 2 -------------------           
    end
    % ------------------- END PARAMETER 1 ---------------------------------
    Outlier_subject_alpha_t{t_a,a_0, d} = Outlier_matrix;   
    TPR_subject_cum_alpha(:,:,d) = TPR_cum;
    FPR_subject_cum_alha(:,:,d) = FPR_cum;
end


%% Find optimal value

clear C
for i = 1:length(Data)
    TPR_vec = TPR_subject_cum(i,:);
    FPR_vec = FPR_subject_cum(i,:);
    C = sqrt((1 - TPR_vec).^2 + FPR_vec.^2);
    [value index_min] = min(C);
    
    TPR_opt(i) = TPR_vec(index_min);
    FPR_opt(i) = FPR_vec(index_min);
    e_opt(i) = e_max(index_min);
end 

TPR_opt = TPR_opt';
FPR_opt = FPR_opt';
e_opt = e_opt';

subjects = [02 05 07 10 13 14 16 20 21 22]; 
legend_text = {'e_max: 0.1', 'e_max: 0.15', 'e_max: 0.2', 'e_max: 0.25', 'e_max: 0.3', 'e_max: 0.35', 'e_max: 0.4', 'e_max: 0.45', 'e_max: 0.50', 'e_max: 0.55', 'e_max: 0.6'};
figure()
subplot(2,1,1)
bar(TPR_subject_cum)
legend(legend_text)
ylabel('Sensitivity')
xlabel('Subject')
xticks(1:10)
xticklabels(subjects)
axis([0 11 0 1])
subplot(2,1,2)

bar(1 -FPR_subject_cum)
legend(legend_text)
ylabel('Specificity')
xlabel('Subject')
xticks(1:10)
xticklabels(subjects)
axis([0 11 0 1])