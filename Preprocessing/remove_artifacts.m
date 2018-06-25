function eegcleaned = remove_artifacts(EEGICA)
% This is from Laura example use.

% load the classifier using spatial features and the mean and standard deviations of the spatial features
load('spatial_established_features3.mat')

% extract features (runs a dipole fit, which is time consuming)
[ic_feats, eegout, speccomp, feature_names] = ic_feature_extraction(EEGICA, {'established_spatial_features'});

% subtract the mean loaded with the classifier from the calculated features
featsnorm = ic_feats - repmat(mu1, size(ic_feats,1),1);

% divide by the standard deviation loaded with the classifier
featsstand = featsnorm./repmat(sigma1, size(ic_feats,1),1);

% use the loaded model to predict probabilities of class memberships of ICs
predprob = mnrval(mod1, featsstand);

% assign each IC to the class with highest probability
[temp1, predclass] = max(predprob, [], 2);

% find blinks, heartbeat, lateral eye movements, and muscle artifacts
artids = [1, 3:5];
intsec = intersect(predclass, artids);
EEGICA.reject.classtype = predclass;

artstoremove = [];
% each IC that is in one of the classes blinks, heartbeat, lateral eye movements, and muscle artifacts
for artid = intsec'
    arts = find(predclass == artid);
    % show ics that will be removed
    if length(arts)==1
    	figure; 
    end
    pop_topoplot(EEGICA, 0, arts, num2str(artids))
    % store ICs to be removed
    artstoremove = [artstoremove arts'];
end

eegcleaned = pop_subcomp(EEGICA, artstoremove, 0);


end 
