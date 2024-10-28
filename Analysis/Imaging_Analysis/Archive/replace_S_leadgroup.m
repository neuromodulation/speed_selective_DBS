%% Copy stimulation parameters
M_old = load('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset_main\derivatives\leadgroup\test\dataset-LeadDBSDataset_analysis-test.mat').M;
M = load('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\dataset-LeadDBSDataset_analysis-test.mat').M; 

%% Replace
M.M.S = M_old.M.S; 

%% Replace model
for i=1:length(M.M.S)
    M.M.S(i).model = M.M.vatmodel;
end

%% Save lead group file
save('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\dataset-LeadDBSDataset_analysis-test_new.mat', "M"); 


