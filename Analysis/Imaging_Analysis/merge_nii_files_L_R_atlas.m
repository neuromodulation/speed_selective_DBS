%% Prepare environemnt
clear all;

%% Load the bounding box 
path0 = "C:\Users\ICN\Documents\Try_without_Sdrive\atlas\bb.nii";
nii0 = ea_load_nii(char(path0));

% Set path where to find nifti files
path_targets = "C:\Users\ICN\Documents\Try_without_Sdrive\atlas\";
files = dir(path_targets);

% Nuceli of interest
targets = ["STN"];

% Loop over structures 
for i=1:length(targets)
    target = string(targets(i));
    % Get the nifti files for left and right
    files_lr = [];
    for i_file=1:length(files)
        name = files(i_file).name;
        if startsWith(name, target) && (endsWith(name, "l.nii") || endsWith(name, "r.nii"))
            files_lr = cat(1, files_lr, path_targets+name);
        end
    end

    % Merge files into new nifti file
    new_file = char(files_lr(1));
    output_path = strcat(new_file(1:end-8),'.nii');
    nii1 = ea_load_nii(char(files_lr(1)));
    nii2 = ea_load_nii(char(files_lr(2)));
    flags =struct();
    flags.mask = -1;
    spm_imcalc(char(nii0.fname, nii1.fname, nii2.fname), char(output_path), 'i2+i3', flags); 

    % Check
    nii3 = ea_load_nii(char(output_path));
    disp(target)
    disp(sum(nii3.img, "all"))
    disp(sum(nii1.img, "all")+sum(nii2.img, "all"))
end