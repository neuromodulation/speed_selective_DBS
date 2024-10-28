%% Prepare environemnt
clear all;

%% Load a functional connectivity nift file (use dimension/voxel size)
path0 = ['C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\' ...
    'derivatives\leaddbs\sub-3\stimulations\MNI152NLin2009bAsym\gs_test\' ...
    'sub-3_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgR_funcmap.nii'];
%reslice_nii(path0_old, path0_new, [2,2,2])
nii0 = ea_load_nii(char(path0));

%% Set path where to find nifti files
path_targets = "C:\Users\ICN\Documents\Try_without_Sdrive\atlas\";
files = dir(path_targets);

% Get regions to merge
filenames = uigetfile(path_targets,'MultiSelect','on');

% Ask for name of new parcellation 
new_name = input("Name:");

% Loop over structures 
equation = "";
paths = [""];
paths(1) = string(nii0.fname);
for i=1:length(filenames)

    % Load nii file
    nii = ea_load_nii(char(path_targets+filenames(i)));
    disp(nii.dim)

    % Store
    if i == 1
        equation = append(equation, char("i"+(i+1)));
    else
        equation = append(equation, char("+i"+(i+1)));
    end
    paths(i+1) = string(nii.fname);
end

%% Merge
output_path = strcat(path_targets+new_name,'.nii');
flags =struct();
flags.mask = -1;
spm_imcalc(squeeze(char(paths)).', char(output_path), char(equation), flags); 

%% Load and check
nii_new = ea_load_nii(char(output_path));

