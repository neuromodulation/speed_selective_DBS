%% Calculate step wise regression 

clear all;

% Set parameters
med = "Off";
Fz = true;
feature_name = "mean_speed";
mode = "mean";
method = "mean";
n_norm = 5;
n_cutoff = 5;


% Load matrix containing the outcome measure
y = load(sprintf("../../../Data/%s/processed_data/res_%s_%s_%s_%d_%d.mat", med, feature_name, mode, method, n_norm, n_cutoff));

% Delete subject 3 (different electrode type) and choose the recovery block
subjects = 1:24;
subjects(4) = []; % Remove subject 3 (index 4 in MATLAB since MATLAB indexing starts at 1)
y = y.res(subjects, 2); % Choose the second column (recovery block)

% Load the mat files containing the connectivity values
if Fz
    matrix_filename = sprintf("../../../Data/%s/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat", med);
else
    matrix_filename = sprintf("../../../Data/%s/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat", med);
end

data = load(matrix_filename);
conn_mat_original = data.X;
seeds = data.seeds;

n_targets = size(conn_mat_original, 1) - length(y);
targets = seeds(1:n_targets);

X = conn_mat_original(1:n_targets, n_targets+1:end)';

% Remove 2 regions
idx_SNc = 4;
idx_inf = 11;
X(:, [idx_SNc, idx_inf]) = [];
targets([idx_SNc, idx_inf]) = [];

%% Try stepwise regression
stepwiseModel = stepwiselm(X, y);%, 'Upper', 'linear', 'PRemove', 0.05, 'Verbose', 1);

% Display the selected features
selectedFeatures = stepwiseModel.PredictorNames;
disp('Stepwise regression');
disp(targets(selectedFeatures));
