%% Load lead Group file
path = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test.mat'];
M = load(path).M;

%% Load file containing the variables of interest
file = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\processed_data\res_peak_speed_cut_5_norm_5_smooth_5_Median.mat'];
variable = load(file).res;
% Select stimulation block 
variable = variable(:, 1);
% Discard patient # 5
variable(4) = [];

%% Replace values 
M.clinical.vars{1} = variable;

%% Save new lead group file 
path_new = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\LeadDBSDataset\derivatives\leadgroup\test\'... 
    'dataset-LeadDBSDataset_analysis-test_new.mat'];
save(path_new,'M','-v7.3');