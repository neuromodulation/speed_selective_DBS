clear all;
%% Prepare the environment 

addpath('C:\code\wjn_toolbox');
%addpath(genpath('C:\code\lead'));
%addpath(genpath('C:\code\spm12'));

%% Define VTA files and variable of interest

resultfig = ea_mnifigure; % Create empty 3D viewer figure

M.pseudoM = 1; % Declare this is a pseudo-M struct, i.e. not a real lead group file

template_path = {['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\LeadDBSDataset\derivatives\leaddbs\sub-'], 
    ['\stimulations\MNI152NLin2009bAsym\gs_test\sub-22_sim-binary_model-simbio_hemi-L.nii']};
ID_list = [2:4 6:25];
nifti_list = cell(length(ID_list), 1);
for i = 1:length(ID_list)
    %nifti_list{i} = sprintf(template_path, ID_list(i));
    if ID_list(i) < 10
        nifti_list{i} = strcat(template_path{1}, '0', string(ID_list(i)), template_path{2});
    else
        nifti_list{i} = strcat(template_path(1), string(ID_list(i)), template_path(2));
    end
end
M.ROI.list = nifti_list;

M.ROI.group=ones(length(M.ROI.list),1);

%% Load variable of interest
file = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\processed_data\res_peak_speed_cut_5_norm_5_smooth_5_Median.mat'];
variable = load(file).res;
% Select stimulation block 
variable = variable(:, 1);
% Discard patient # 5
variable(4) = [];
M.clinical.labels={'DiffFastSlow'}; % how will variables be called
M.clinical.vars{1} = variable;

M.guid='Analysis'; % give your analysis a name

save('Analysis_Input_Data.mat','M'); % store data of analysis to file

%% Open up the Sweetspot Explorer
ea_sweetspotexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);

% Open up the Network Mapping Explorer
%ea_networkmappingexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);

% Open up the Fiber Filtering Explorer
ea_discfiberexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);





