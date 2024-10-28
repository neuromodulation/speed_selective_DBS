%% Extract nifti files from atlases based on an excel sheet
clear all

%% open the txt file specifiying the targets and the atlases
ROIs = readtable('extract_areas_atlas.xlsx');

% Specify path fo atlases
atlas_path = 'C:\Users\ICN\Documents\Try_without_Sdrive\atlas\';

%% Loop over atlases
for i=1:height(ROIs)
    
    % Load the atlas nifti file from which you want to extract
    atlas_filename = string(ROIs{i, 1});
    atlas_file_path = char(strcat(atlas_path, strcat(atlas_filename, ".nii")));
    atlas_nii = ea_load_nii(atlas_file_path);
    nii_original = atlas_nii.img;
    % Load the corresponding text file
    text = readtable(strcat(atlas_file_path(1:end-4),'.txt'));

    % Loop over the targets to extract from one atlas 
    targets = ROIs{i, 2:end};
    for j=1:width(targets)
        target = char(targets{1, j});

        if target ~= "none"
            % Loop over the text file and find all corresponding IDs
            idx = find(contains(text{:, 2}, target));
            IDs = text{idx, 1};
    
            % Save as one nifti file 
            nii = atlas_nii;
            new_img = ismember(nii_original, IDs);
            nii.img = new_img;
            nii.fname = strcat(atlas_path, target,'.nii');
            ea_write_nii(nii)
    
            % Save for left
            ID_names = text{idx, 2};
            IDs_left = IDs(endsWith(ID_names, "_L") | startsWith(ID_names, "Left"));
            new_img = ismember(nii_original, IDs_left);
            nii.img = new_img;
            nii.fname = strcat(atlas_path, target,'_L.nii');
            ea_write_nii(nii)
    
            % Save for right
            IDs_right = IDs(endsWith(ID_names, "_R") | startsWith(ID_names, "Right"));
            new_img = ismember(nii_original, IDs_right);
            nii.img = new_img;
            nii.fname = strcat(atlas_path, target,'_R.nii');
            ea_write_nii(nii)
        end
    end
end