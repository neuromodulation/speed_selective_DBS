%% Settings
connectome_type = '_seed-fMRI_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii';
atlas_path = 'DISTAL_NANO.nii';
%atlas_path = 'compound_atlas_HCPex_SUIT_ABGT.nii';
%subjects = [2:4 5:25];
subjects = [2:4 6:25];
subject_path = ["C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\" + ...
    "derivatives\leaddbs\sub-", "\stimulations\MNI152NLin2009bAsym\gs_test\"];

%% Generate the fMRI connectivity maps

%lead_mapper
% 1: Seed and Output definition - "Manually choose seeds" (use STN-LFP contacts)
% 2: Connectome definition - "PPMI 74_15 (Horn 2017) > Patients"
% 3: Command - "Connectivity map from seed"


%% Load the atlas and corresponding text file
atlas_table = readtable([atlas_path(1:end-4), '.txt']);
atlas_table.Properties.VariableNames = {'Index', 'Name'};
atlas_nii = ea_load_nii(atlas_path);

%% Calculate functional connectivity

% Store result variable_names = {'Subject ID', 'ROI', 'Value'};
res = array2table(zeros(0,3),'VariableNames',variable_names);

% Loop over every subject
for i = 1:length(subjects)
    
    % Define the name of the patient as present in the folder
    sub_id = string(subjects(i));

    % Find seed connectome file
    connectome_path = subject_path(1) + sub_id + subject_path(2);
    fnames = dir(connectome_path);
    for file_i = 1:length(fnames)
        name = fnames(file_i).name;
        if endsWith(name, connectome_type)
            connectome_fname = name;
            break
        end
    end

    % What does spm_imcalc do? I think making a mask based on the atlas
    % (with the regions of interest) and applying it to the functional
    % connectivity map
    connectome_file_path = char(connectome_path+connectome_fname);
    spm_imcalc({atlas_path, connectome_file_path},...
        'temp_corrected_roi_file.nii', 'i2'); % ensure files have same dimensions
    con_nii = ea_load_nii('temp_corrected_roi_file.nii');
    
    % Loop over all the regions in the atlas 
    for roi_i = 1:length(atlas_table.Index)
        % Calculate average connectivity for region of interest (ROI)
        con_value = nanmean(con_nii.img(atlas_nii.img == atlas_table.Index(roi_i))); % average connectivity at ROI
        % Store as new row in result table 
        target_name = atlas_table.Name{roi_i};
        tmp = array2table([sub_id, target_name, con_value], "VariableNames", variable_names);
        res = vertcat(res, tmp);
    end
end

%% Save results
writetable(res, 'fmri_connectivity_maps_Fz.xlsx');
