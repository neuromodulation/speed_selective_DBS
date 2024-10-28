%% Prepare environemnt
clear all;

%% Load the bounding box 
path0 = "C:\Users\ICN\Documents\Try_without_Sdrive\atlas\bb.nii";
nii0 = ea_load_nii(char(path0));

% Set path where to find nifti files
paths_1 = ["C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leaddbs\sub-", "\stimulations\MNI_ICBM_2009b_NLIN_ASYM\MNI152NLin2009bAsym\gs_test\"];
paths_2 = ["C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leaddbs\sub-", "\stimulations\MNI152NLin2009bAsym\gs_test\"];

% Array of patients of interest
subjects = [2:4 6:25];

% Loop over patients 
for i=1:length(subjects)
    sub_id = string(subjects(i));
    path_sub = paths_1(1) + sub_id + paths_1(2);
    files = dir(path_sub);
    % Try different path
    if isempty(files)
        sub_id = string(subjects(i));
        path_sub = paths_2(1) + sub_id + paths_2(2);
        files = dir(path_sub);
    end
    % Get the nifti files for left and right VTA
    files_lr = [];
    for i_file=1:length(files)
        name = files(i_file).name;
        if endsWith(name, 'sim-binary_model-simbio_hemi-L.nii') || endsWith(name, 'sim-binary_model-simbio_hemi-R.nii')
            files_lr = cat(1, files_lr, path_sub+name);
        end
    end

    % Merge files into new nifti file
    new_file = char(files_lr(1));
    output_path = strcat(new_file(1:end-11),'.nii');
    nii1 = ea_load_nii(char(files_lr(1)));
    nii2 = ea_load_nii(char(files_lr(2)));
    flags =struct();
    flags.mask = -1;
    spm_imcalc(char(nii0.fname, nii1.fname, nii2.fname), char(output_path), 'i2+i3', flags); 

    % Check
    nii3 = ea_load_nii(char(output_path));
    disp(sub_id)
    disp(sum(nii3.img, "all"))
    disp(sum(nii1.img+nii2.img, "all"))
end