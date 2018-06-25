function [FPR, TPR, f] = ROC_points(E_train, E, seizure,Data_train_clean)
% This code is based on Algortihm 1 from An introductoin to ROC analysis by
% Tom Fawcett

% Start by getting the score f (log(p))
% log_p = log_normal(E_train, E);
[log_p, log_p_train, log_p_test] = log_normal(E_train, E);

if Data_train_clean
    f = log_p_test;
else
    f = log_p;
end
% Start by making a vector L that indicates if point is a true outlier or not 
L = zeros(1,length(f));
L(seizure) = 1; 

% Sort L based on the score 
[~,f_sort] = sort(f);
L_sort = L(f_sort);

% initilize paramters
TP = 0; % counter
FP = 0; % couner 

% True posiitive rate (TPR): True positive / Condition positive
% False positive rate (FPR): False positive / Conditoin negative
FPR = [];
TPR = [];
P = length(seizure);
N = length(f) - P;

f_prev = min(f)-100; % lowest threshold
i  = 1; 

% We only need to do this until all the outliers have been found 
iter = find(L_sort,1, 'last');
if iter < length(L_sort)
    iter = iter + 1;
end
while i <= length(L_sort)%iter
    if f(i) ~= f_prev
        FPR = [FPR FP/N];
        TPR = [TPR TP/P];
        f_prev = f(i);
    end 
    if L_sort(i) == 1 
        TP = TP + 1;
    else
        FP = FP + 1;
    end 
    i = i + 1;
end 

FPR = [FPR 1];
TPR = [TPR 1];   
        
end 