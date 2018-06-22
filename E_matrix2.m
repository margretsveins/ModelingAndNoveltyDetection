function [E, E_train, explained] = E_matrix2(Data, Data_train, Data_test,length_window,varargin)
% PCA - Test and Train
% This function runs PCA on training data. Then it transform all the data
% by using the principal compent coefficant. Afterwards it calculates the
% energy for each PC in each window. Energy is defined as 
%                   e = sum(Y.^2)
% Input:
%       EEG_clean: EEG data
%       W: The Window matrix for the EEG data 
%       size_train: Fraction that say how big the training set should be
%       num_window: How many window the data was split to
%       srate: How many point are in each window
%       run: This tell us what data we are using (used for titles on plots)
%       component: Boolean. True if we only want to use the PC that explain
%           the 99% of the data. False if we want to use all the PC
% varargin
%       PC: Boolean. True if we only want to use the PC that explain
%           the 99% of the data. False if we want to use all the PC
%       Plot: Boolean. If true then we plot the distribution.
% Output: 
%       E: The energy matrix
%       E_train: The training part of the energy matrix

% Get varargin
% only want 5 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 7
    error('PCA_TrainTest requires at most 6 optional inputs');
end
% set defaults for optional inputs (no plot and use all the PC)
optargs = {'Plot' false 'Missing title', 'Standardize', false};
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[Plot, Plot_boleen, Plot_title, standardize, standardize_boleen] = optargs{:};


% Create windows
[W, num_window] = Frames(Data, length_window);

% Start to declare number of test and training window.
num_train_w = floor(length(Data_train)/length_window);
num_test_w = num_window - num_train_w;

% Make the training matrix
clean_data = Data;
[num_var num obs] = size(clean_data);
W_train = Data_train;
W_test  = Data_test;

% Run PCA on the training matrix
% pca(X) returns the principal components coefficients for the n by p data
% matri X. Rows of X correspond to observations and columns correspond to
% variables. The coefficient matrix is p by p matrix. Each column of coeff
% contains coefficients for one principal componnets, and the columns are
% in descending order of components variance. 
% The w matrix is structured so each row correspond to variables and each
% column to observations so we need to transpose w before we run PCA

[U, Y_trans, latent, tsquared, explained] = pca(W_train', 'Centered', false);
% Transform the training and test windows and calculate the energy e
E = zeros(num_var,num_window);


for i = 1:num_window
    X = W(:,:,i);
    Y = U'*X;
    %e  = Y';
    e = sum(Y'.^2);
    E(:,i) = e';
end 
    E_train = E(:,1:num_train_w);
    E_test = E(:,num_train_w+1:end);    


if Plot_boleen
    [E_test_sample test_index] = datasample(E_test',num_train_w);
    E_eq_TestTrain = [E_train E_test_sample']; 

    % Scatterplot with histogram 
     test_train = {};
     for i = 1:2*num_train_w
         if i < num_train_w+1
             test_train{i,1} = 'Train';
         else
             test_train{i,1} = 'Test';
         end 
     end  

    figure()
    scatterhist(E_eq_TestTrain(1,:),E_eq_TestTrain(2,:),'Group',test_train,'Kernel','on')
    title(['Train: ' num2str(size_train*100) '% for run:' Plot_title])
end 
end 

