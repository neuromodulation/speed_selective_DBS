%% Plot the stimulation probability (averaged across hemispheres)
clear all; 
close all;

%% Set path where to find nifti files
paths = ["C:\Users\ICN\Documents\Try_without_Sdrive\LeadDBSDataset\derivatives\leaddbs\sub-", "\stimulations\MNI152NLin2009bAsym\gs_test\"];

% Array of patients of interest
subjects = [2:4 6:25];
n_subjects = length(subjects); 

% Loop over patients 
files_l = [];
files_r = [];
for i=1:length(subjects)
    sub_id = string(subjects(i));
    path_sub = paths(1) + sub_id + paths(2);
    files = dir(path_sub);

    % Get nii files of VTA
    for i_file=1:length(files)
        name = files(i_file).name;
        if endsWith(name, 'sim-binary_model-simbio_hemi-L.nii')
            files_l = cat(1, files_l, path_sub+name);
        end
        if endsWith(name, 'sim-binary_model-simbio_hemi-R.nii')
            files_r = cat(1, files_r, path_sub+name);
        end
    end 
end

% Sum nifti files
files_lr = [files_l, files_r];
output_paths = ["C:\Users\ICN\Documents\Try_without_Sdrive\sum_l.nii", "C:\Users\ICN\Documents\Try_without_Sdrive\sum_r.nii"];

% Generate expression
expression = '';
for i = 1:n_subjects
    expression = [expression, 'i', num2str(i)];
    expression = [expression, ' + '];
end
expression = expression(1:end-3); 

for i=1:2
    files = files_lr(:, i);
    spm_imcalc(char(files), char(output_paths(i)), expression); 
end

%% Flip nii from left to right
fname_l_flipped = 'C:\Users\ICN\Documents\Try_without_Sdrive\sum_l_flipped_r.nii';
ea_flip_lr(char(output_paths(1)), fname_l_flipped);

%% Average 
fname_r = 'C:\Users\ICN\Documents\Try_without_Sdrive\sum_r.nii';
output_path = 'C:\Users\ICN\Documents\Try_without_Sdrive\sum_lr_flags.nii';
fname_l_flipped = 'C:\Users\ICN\Documents\Try_without_Sdrive\sum_l_flipped_r.nii';
flags = struct();
flags.mask = 0;
flags.dmtx = 0;  % Data is not matrix, leave as default
flags.interp = 1;  
spm_imcalc({fname_r, fname_l_flipped}, output_path, '(i1+i2)/2', flags); 

%% Plot the STN
ea_mnifigure;
color = "white";
alpha = 0.4;
load([ea_space([],'atlases'),'DISTAL Nano (Ewert 2017)',filesep,'atlas_index.mat']); % manually load definition of DISTAL atlas.
rSTN=atlases.roi{4,1}.fv;
patch('Faces',rSTN.faces,'Vertices',rSTN.vertices,'facecolor',color,'edgecolor',color, 'facealpha', alpha, 'edgealpha', 0);
lSTN=atlases.roi{4,2}.fv;
patch('Faces',lSTN.faces,'Vertices',lSTN.vertices,'facecolor',color,'edgecolor',color, 'facealpha', alpha, 'edgealpha', 0);

output_path = 'C:\Users\ICN\Documents\Try_without_Sdrive\sum_lr.nii';
thresholds = [13, 9, 5, 1];
alpha = 0.8;
for i=1:length(thresholds)
    nii = ea_load_nii(output_path);
    img_new = nii.img; 
    img_new(img_new > thresholds(i)) = 100;
    nii.img = img_new;
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',alpha, 'edgealpha', 0);
    alpha = alpha - 0.2;
end




