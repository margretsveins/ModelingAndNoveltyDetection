function EEG_clean = cleanData(EEG_rec)
% Run ICA and remove artifacts on rec data 

% Run ICA to obtain ICA decomposition for dataset
EEGICA_rec = pop_runica(EEG_rec, 'extended',1,'interupt','off'); 

% Remove artifacts 
EEG_clean = remove_artifacts(EEGICA_rec); % <-- FUNCTION 

end 