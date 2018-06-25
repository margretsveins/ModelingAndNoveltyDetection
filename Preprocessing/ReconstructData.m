function EEG_rec = ReconstructData(EEG_uni)
%% Run PCA due to ranking issues in the data 
srate = EEG_uni.srate;
% num_sek = 1200;
% length_testdata = srate*num_sek;
data_uni = EEG_uni.data;
rank_data = rank(data_uni); % rank = 17 
[U_uni, Y_uni, latent_uni, tsquared_uni, explained_uni] = pca(data_uni', 'Centered', false);
Y_uni = Y_uni';

% We want to reduce the variables, we will select the PC that explan 99% of
% the variance in the data.
num_prin = rank(data_uni);
while sum(explained_uni(1:num_prin)) < 99
    num_prin = num_prin + 1;
end 

% Set all the PC coefficent to zero that we want to get rid off 
U_uni(:,num_prin+1:end) = 0;

% Reconstructed data 
Y_rec = U_uni'*data_uni;
data_uni_rec = U_uni*Y_rec;
rank_data_rec = rank(data_uni_rec);

% Make a new EEG set with the reconstructed data 
EEG_rec = EEG_uni;
EEG_rec.data = data_uni_rec;


end 