%% Load lead Group file

%% Load file containing variables
file = ['C:\Users\ICN\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor\' ...
    'vigor_stim_task\Data\Off\processed_data\res_peak_speed_cut_5_norm_5_smooth_5_Median.mat'];
variable = load(file).res;
% Select stimulation block 
variable = variable(:, 1);
% Discard patient # 5
variable(4) = [];