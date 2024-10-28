% Load the Lead group file 
clear all;
path = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test.mat'];
M = load(path).M;

%% Get new order
list = M.patient.list; 
IDs = [];
% Find new order 
for i=1:length(list)
    tmp = split(list{i}, "-");
    ID = str2num(tmp{end});
    IDs = [IDs ID];
end
[ordered_IDs,reorder] = sort(IDs);

%% New M 
newM=M;
newM.patient.list=M.patient.list(reorder);
newM.patient.group=M.patient.group(reorder);
newM.isomatrix{1}=M.isomatrix{1}(reorder);
newM.elstruct=M.elstruct(reorder);
newM.stats=M.stats(reorder);
newM.S=M.S(reorder);
for varnum=1:length(M.clinical.vars)
newM.clinical.vars{varnum}=M.clinical.vars{varnum}(reorder);
end

M=newM;

%% Optional replace stimulation parameters
% path = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\'... 
%     'dataset-LeadDBSDataset_analysis-test_old2.mat'];
% M2 = load(path).M;
% newM.S = M2.S;


%% Save 
path_new = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test_new.mat'];
save(path_new,'M','-v7.3');
