%% Prepare
clear all; 

%% Create array storing all files of interest
mode = "all"; % Define which mode: R(right), L(eft) or all
paths_all = [];

%% Add the regions of interest
ROI_path = 'C:\Users\ICN\Documents\Try_without_Sdrive\atlas\';
if mode == "L"
    filenames = uigetfile(strcat(ROI_path,'*_L.nii'),'MultiSelect','on');
elseif mode == "R"
    filenames = uigetfile(strcat(ROI_path,'*_R.nii'),'MultiSelect','on');
else
    filenames = uigetfile(strcat(ROI_path,'*.nii'),'MultiSelect','on');
end
for i=1:length(filenames)
    filename = char(filenames(i));
    paths_all = cat(1, paths_all, string(strcat(ROI_path,filename)));
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
        if (endsWith(name, 'sim-binary_model-simbio.nii') & mode == "all") ||... 
            (endsWith(name, 'sim-binary_model-simbio_hemi-R.nii') & mode == "R") ||...
            (endsWith(name, 'sim-binary_model-simbio_hemi-L.nii') & mode == "L") 
            paths_all = cat(1, paths_all, path_sub+name);
        end
    end
end

%% Write in text file 
fileID = fopen(sprintf('connectivity_matrix_sources_%s.txt',mode), 'w');

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