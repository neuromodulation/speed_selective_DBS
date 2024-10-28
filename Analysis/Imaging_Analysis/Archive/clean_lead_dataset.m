%% Clean LeadDBS dataset

% Run through folder and delete everything with simbio in the name 
rootdir = "C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset - Copy";
filelist = dir(fullfile(rootdir, '**\*.*'));

for i=1:length(filelist)
    if contains(filelist(i).name,"simbio")
        disp(filelist(i).name)

        % Delete file 
        file_path = string(filelist(i).folder) + "\"+ string(filelist(i).name);
        delete(file_path);
    end
end

%% Rename lead group stimulation 
M = load('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\analysis1\dataset-LeadDBSDataset_analysis-analysis1.mat');
M.root = 'C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\analysis1\';
M.guid = 'analysis1';
%% Rename stimulation label 
for i=1:length(M.S)
    M.S(i).label = 'stim1_ossdbs';
end

%% Replace S 
clear all;
M1 = load('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDatasetOLD\derivatives\leadgroup\analysis1\dataset-LeadDBSDataset_analysis-analysis1.mat').M;
M = load('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\a1\dataset-LeadDBSDataset_analysis-a1.mat').M;
old_label= M.S.label;
M.S = M1.S;
for i=1:length(M.S)
    M.S(i).label = old_label;
end

% Change order 
M.S = [M.S(1:3), M.S(24), M.S(4:23)];
%% Save
save('C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\analysis1\dataset-LeadDBSDataset_analysis-analysis1.mat');
