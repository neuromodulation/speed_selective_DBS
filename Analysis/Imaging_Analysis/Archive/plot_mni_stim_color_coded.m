%% Plot the center of the stimulation electrode in the STN color coded by a variable 
%clear all; 
%close all;
addpath(genpath("C:\CODE\ac_toolbox\"))

%% Load variable of interest 
variable_path = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - " + ...
    "PROJECT ReinforceVigor\vigor_stim_task\Data\Off\processed_data\" + ...
    "res_mean_speed_mean_mean_5_5.mat";
variable = load(variable_path);
variable = variable.res(:, 2);
variable(4) = [];
%variable(16) = 0;

%% Define the colors according to the variable
% Make sure the maximum absolute values in the positive and negative
% dimension are the same
neg_id = variable < 0;
abs_variable = abs(variable);
norm_variable = normalize(abs_variable, 'range', [0, 0.5]);
norm_variable(neg_id) = norm_variable(neg_id) * -1;
norm_variable = norm_variable + 0.5; 

% choose colormap and colors according to normalized values
n = 256;
cmap = redblue(n); 
color_indices = round(norm_variable * (n - 1)) + 1;
colors = cmap(color_indices, :);
colorbar
save("..\Imaging_Analysis\colors_mni.mat", 'colors')
%% Load csv file containing mne coordinates
info_path = "C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - " + ...
    "PROJECT ReinforceVigor\vigor_stim_task\Data\Dataset_list_Off.xlsx";
info = readtable(info_path);
hand = info([2:4 6:25], :).Hand;

% Define subject of interest
subjects = [2:25];
subjects = [2:4 6:25]

%% Open a mnifigure
%figure = ea_mnifigure;

%% Add the STN on both sides 
color = "white";
alpha = 0.2;
%ea_mnifigure;
load([ea_space([],'atlases'),'DISTAL Nano (Ewert 2017)',filesep,'atlas_index.mat']); % manually load definition of DISTAL atlas.
rSTN=atlases.roi{4,1}.fv;
patch('Faces',rSTN.faces,'Vertices',rSTN.vertices,'facecolor',color,'edgecolor',color, 'facealpha', 0, 'edgealpha', alpha);
lSTN=atlases.roi{4,2}.fv;
patch('Faces',lSTN.faces,'Vertices',lSTN.vertices,'facecolor',color,'edgecolor',color, 'facealpha', 0, 'edgealpha', alpha);
% Set camera view
% v.az= -107.2793;
% v.el= -11.0918;
% v.camva= 1.3153;
% v.camup= [0.019811478687700,-0.008624067118847,0.999766538137047];
% v.camproj= 'orthographic';
% v.camtarget= [0,-18,9.999999999999988];
% v.campos= [-1.429900473319951e+03,6.044450910288658e+02,43.704319668799600];
%ea_view(v); 

%% Add the STN sweetspot (Horn et al, Caire et al)
% mni_sp_R = [12.42 -12.58 -5.92];
% mni_sp_L = [-12.58 -13.41 -5.81];
% color = "yellow";
% alpha = 0.7;
% radius = 0.5;
% %plot_mni_roi_adjusted(mni_sp_R, color, alpha, radius);
% plot_mni_roi_adjusted(mni_sp_L, color, alpha, radius);

%% Add the center of the electrode ring used for stimulation color coded by the varbiable
alpha = 0.7; 
radius = 0.5;
% Loop over subjects 
for i=1:length(subjects)
    sub_id = string(subjects(i));
    subject_row = info(string(info.NumberOfParticipant) == sub_id, :);

    % Get the MNI coordinates
    mni_R = [subject_row.MNI_stim_R_x, subject_row.MNI_stim_R_y, subject_row.MNI_stim_R_z];
    mni_L = [subject_row.MNI_stim_L_x, subject_row.MNI_stim_L_y, subject_row.MNI_stim_L_z];
    %mni_mean = mean([mni_R; mni_L], 1);

    if hand(i) == "L"
        mni_R_copy = mni_R;
        mni_L_copy = mni_L;
        mni_R = mni_L_copy; 
        mni_L = mni_R_copy;
    end

    % Plot rights
    color = colors(i, :);
    plot_mni_roi_adjusted(mni_R, color, alpha, radius);
    % 
    % Plot left
    %color = "yellow"
    plot_mni_roi_adjusted(mni_L, color, alpha, radius);

    % Plot mean
    %plot_mni_roi_adjusted(mni_mean, color, alpha, radius);
end


