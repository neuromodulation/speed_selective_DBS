% Add lead MNI coordinates to BIDS file 

% Define subject of interest
sub = "L017";

% Read the target BIDS tsv file
root = sprintf(['C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\' ...
    'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys\\' ...
    'rawdata\\sub-%s\\*.tsv'], sub);
[file,path] = uigetfile(root, 'All files');

% Read file 
t = readtable(strcat(path,file), "FileType","text",'Delimiter', '\t');

% Read the ea_reconstruction file containing the coordinates
reco = load(sprintf(['C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\' ...
    'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys\\' ...
    'derivatives\\LFP_leads\\%s\\ea_reconstruction.mat'], sub));
coords_right = reco.reco.mni.coords_mm{1, 1};
coords_left = reco.reco.mni.coords_mm{1, 2};

% Insert coddinates in target BIDS file 
t_new = t; 
t_new(1:8, 2:4) = array2table(coords_right);
t_new(9:16, 2:4) = array2table(coords_left);

% Save updated BIDS file (for all coordinate files for that subject)
root = sprintf(['C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\' ...
    'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys\\' ...
    'rawdata\\sub-%s\\'], sub);
filelist = dir(fullfile(root, '**\*.tsv'));  %get list of files and folders in any subfolder
for i = 1:length(filelist)
    if contains(filelist(i).name, "MNI")
        writetable(t_new, strcat(filelist(i).folder,'\\',filelist(i).name), 'filetype','text', 'delimiter','\t')
        disp("Updated")
    end
end

% Also 