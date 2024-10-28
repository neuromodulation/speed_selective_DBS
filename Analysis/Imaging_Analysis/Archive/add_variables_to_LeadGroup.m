%% Load lead Group file
path = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\a1\'... 
    'dataset-LeadDBSDataset_analysis-a1.mat'];
M = load(path).M;
path_backup = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\a1\old\'... 
    'dataset-LeadDBSDataset_analysis-a1.mat'];
save(path_backup,'M','-v7.3');

%% Load files containing the variables of interest
path_matfiles = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - ' ...
    'PROJECT ReinforceVigor\vigor_stim_task\Data\Off\processed_data\'];
filenames = uigetfile(['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - ' ...
    'PROJECT ReinforceVigor\vigor_stim_task\Data\Off\processed_data\*.m;*.mdl;*.mat'],'MATLAB Files','MultiSelect','on');
% Loop over selected mat files
for i=1:length(filenames)
    filename = string(filenames(i));
    variable = load(path_matfiles+filename).res;
    % Select stimulation block 
    variable = variable(:, 2);
    % Discard patient # 5
    %variable(4) = [];

    % Create a new variable in the 
    filename = char(filename);
    M.clinical.labels{1, i} = filename(1:end-4);
    M.clinical.vars{1, i} = variable;
end

%% Save new lead group file 
path_new = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leadgroup\a1\'... 
    'dataset-LeadDBSDataset_analysis-a1.mat'];
save(path_new,'M','-v7.3');