%%

% path1 = "C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset_old\" + ...
%     "derivatives\leaddbs\sub-3\stimulations\MNI152NLin2009bAsym\gs_test\sub-3_sim-binary_model-simbio_hemi-L.nii";
% nii1 = ea_load_nii(char(path1));
% path2 = "C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset_old\" + ...
%     "derivatives\leaddbs\sub-3\stimulations\MNI152NLin2009bAsym\gs_test\sub-3_sim-binary_model-simbio_hemi-R.nii";
% nii2 = ea_load_nii(char(path2));
% 
% flags =struct();
% flags.mask = -1;
% 
% % Add together
% output_path = "C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset_old\" + ...
%     "derivatives\leaddbs\sub-3\stimulations\MNI152NLin2009bAsym\gs_test\test_merge.nii";
% spm_imcalc(char(nii0.fname, nii1.fname, nii2.fname), char(output_path), 'i2+i3', flags); 
% nii3 = ea_load_nii(char(output_path));
% 
% nii3 = nii1;
% nii3.img = nii1.img + nii2.img;
% nii3.fname = char(output_path);
% nii3.pinfo = [1;0;nii3.pinfo(3)];
% ea_write_nii(nii3);

% GPe_l = "C:\CODE\leaddbs\templates\space\MNI152NLin2009bAsym\atlases\DISTAL Nano (Ewert 2017)\lh\GPe.nii.gz\GPe.nii";
% GPe_r = "C:\CODE\leaddbs\templates\space\MNI152NLin2009bAsym\atlases\DISTAL Nano (Ewert 2017)\rh\GPe.nii.gz\GPe.nii";
% GPi_l = "C:\CODE\leaddbs\templates\space\MNI152NLin2009bAsym\atlases\DISTAL Nano (Ewert 2017)\lh\GPi.nii.gz\GPi.nii";
% GPi_r = "C:\CODE\leaddbs\templates\space\MNI152NLin2009bAsym\atlases\DISTAL Nano (Ewert 2017)\rh\GPi.nii.gz\GPi.nii";
% atlas_path = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation -" + ...
%     " PROJECT ReinforceVigor\vigor_stim_task\Code\Analysis\Imaging_Analysis\atlas\derivatives\leaddbs" + ...
%     "\sub-atlas\stimulations\MNI152NLin2009bAsym\gs_test\";
% atlas_path = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation -" + ...
%     " PROJECT ReinforceVigor\vigor_stim_task\Code\Analysis\Imaging_Analysis\atlas\";
% paths_1 = ["C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\" + ...
%     "vigor_stim_task\Data\Off\LeadDBSDataset\derivatives\leaddbs\sub-", "\stimulations\MNI_ICBM_2009b_NLIN_ASYM\MNI152NLin2009bAsym\gs_test\"];
% paths_2 = ["C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\" + ...
%     "vigor_stim_task\Data\Off\LeadDBSDataset\derivatives\leaddbs\sub-", "\stimulations\MNI152NLin2009bAsym\gs_test\"];

%% Load files
% Get atlas of interest (.nii)
atlas_path = ['C:\Users\ICN\Documents\Try_without_Sdrive\atlas\'];
filename = uigetfile([strcat(atlas_path,'*.nii')]);

% load the nifti file from which you want to extract
file_path = strcat(atlas_path, filename);
nii = ea_load_nii(file_path);
% load the text file
text = readtable(strcat(atlas_path, filename(1:end-4),'.txt'));

%% Loop over targets 
IDs = [];
for i_target=1:length(targets)
     target = string(targets(i_target));
    % Find ID in text file 
    idx = find(text{:, 2}== target);
    ID = text{idx, 1};
    IDs = cat(1, IDs, ID);
end

%% Replace entries in img with 0 if not part of IDs (1 otherwise)
nii_original = nii.img;

%% Save left
new_img = ismember(nii_original, IDs(1));
nii.img = new_img;
%change the location and the file name in the fname section
target = char(targets(1));
nii.fname = strcat(atlas_path, target,'.nii');
% save nifti
ea_write_nii(nii)

%% Save right
new_img = ismember(nii_original, IDs(2));
nii.img = new_img;
target = char(targets(2));
nii.fname = strcat(atlas_path, target,'.nii');
ea_write_nii(nii)

%% Save both together
new_img = ismember(nii_original, IDs);
nii.img = new_img;
target = char(targets(1));
nii.fname = strcat(atlas_path, target(1:end-2),'.nii');
%nii.fname = strcat(atlas_path, target(6:end),'.nii');
ea_write_nii(nii)

%% Prepare
clear all; 

%% Create array storing all files of interest
mode = "R"; % Define which mode: R(right), L(eft) or all
paths_all = [];

%% Add the nuclei of interest
targets = ["GPe", "GPi", "Primary_Motor_Cortex", "Putamen", "Supp_Motor_Area"];
atlas_path = "C:\Users\ICN\Documents\Try_without_Sdrive\atlas\";
atlas_files = dir(atlas_path);
for i_target=1:length(targets)
    target = string(targets(i_target));
    for i_file=1:length(atlas_files)
        name = atlas_files(i_file).name;
        if isequal(name, target+".nii")
            paths_all = cat(1, paths_all, atlas_path+name);
        end
    end
end
%% Add the VTA files
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
    % Loop over files and add VTA nifti file
    for i_file=1:length(files)
        name = files(i_file).name;
        if endsWith(name, 'sim-binary_model-simbio.nii')
            paths_all = cat(1, paths_all, path_sub+name);
        end
    end
end

%% Write in text file 
fileID = fopen('connectivity_matrix_sources.txt', 'w');

% Write each string to a separate line
for i = 1:length(paths_all)
    if i == length(paths_all)
        fprintf(fileID, '%s', paths_all(i));
    else
        fprintf(fileID, '%s\n', paths_all(i));
    end
end

% Close file
fclose(fileID);