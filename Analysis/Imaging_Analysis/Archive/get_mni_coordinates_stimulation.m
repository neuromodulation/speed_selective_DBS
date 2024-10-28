% Extract the mni coordinates from the reconstruction files and add only the coordinates
% from the stimulation contact in a excel file for further analysis

%% Load csv file containing the information on the dataset
info_path = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - " + ...
    "PROJECT ReinforceVigor\vigor_stim_task\Data\Dataset_list.xlsx";
info = readtable(info_path,'Sheet','Off');

%% Define subjects of interest
subjects = 1:height(info);
subject_path = ["C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\" + ...
    "derivatives\leaddbs\sub-", "\reconstruction\sub-", "_desc-reconstruction.mat"];

% Define arrays to store the coordinates
mni_stim_L_all = [];
mni_stim_R_all = [];

% Loop over subjects
for i = 1:length(subjects)
    
    if i ==1 || i > 25
        mni_stim_R_all = cat(1, mni_stim_R_all, NaN(1, 3));
        mni_stim_L_all = cat(1, mni_stim_L_all, NaN(1, 3));
    else
    
        % Define the name of the patient as present in the folder
        sub_id = string(subjects(i));
    
        % Load reconstruction file
        file_path = subject_path(1) + sub_id + subject_path(2)+ sub_id + subject_path(3);
        rec = load(file_path);
    
        % Extract info for subject
        subject_row = info(string(info.NumberOfParticipant) == sub_id, :);
        contact_R = string(subject_row.StimulationContactRight);
        contact_L = string(subject_row.StimulationContactLeft);
    
        % Extract coordinates
        mni_R = cell2mat(rec.reco.mni.coords_mm(1));
        mni_L = cell2mat(rec.reco.mni.coords_mm(2));
    
        % Get coordinate of stimulation contact right
        if contact_R == "2,3,4"
            mni_stim_R = mean(mni_R(2:4, :), 1);
        elseif contact_R == "5,6,7"
            mni_stim_R = mean(mni_R(5:7, :), 1);
        elseif contact_R == "7,8,9"
            mni_stim_R = mean(mni_R(7:9, :), 1);
        end
    
        % Get coordinate of stimulation contact left
        if contact_L == "2,3,4"
            mni_stim_L = mean(mni_L(2:4, :), 1);
        elseif contact_L == "5,6,7"
            mni_stim_L = mean(mni_L(5:7, :), 1);
        elseif contact_L == "7,8,9"
            mni_stim_L = mean(mni_L(7:9, :), 1);
        end
        
        % Append 
        mni_stim_R_all = cat(1, mni_stim_R_all, mni_stim_R);
        mni_stim_L_all = cat(1, mni_stim_L_all, mni_stim_L);
    end
end

%% Append to info excel sheet and save 
info.MNI_stim_R_x = mni_stim_R_all(:, 1);
info.MNI_stim_R_y = mni_stim_R_all(:, 2);
info.MNI_stim_R_z = mni_stim_R_all(:, 3);
info.MNI_stim_L_x = mni_stim_L_all(:, 1);
info.MNI_stim_L_y = mni_stim_L_all(:, 2);
info.MNI_stim_L_z = mni_stim_L_all(:, 3);

%% Save 
info_path_new = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - " + ...
    "PROJECT ReinforceVigor\vigor_stim_task\Data\Dataset_list_new.xlsx";
writetable(info, info_path_new);