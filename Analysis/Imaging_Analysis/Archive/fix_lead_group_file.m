%% Fix leas group file --> replace patient folders (that got messed up for soem reason)
clear all;
path = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test.mat'];
M = load(path).M;

%% Get patient list
patient_list = M.patient.list;
target_path = "C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leaddbs\";
% Loop through list and replace with correct filename

for i=1:length(patient_list)
    tmp = patient_list(i); 
    tmp = split(string(tmp), "\");
    patient_list{i} = char(target_path+ tmp(end));
end

%% Optional replace root 
M.root = 'C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\';

%% Replace and save 
M.patient.list = patient_list;
path_new = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test_new.mat'];
save(path_new,'M','-v7.3');
