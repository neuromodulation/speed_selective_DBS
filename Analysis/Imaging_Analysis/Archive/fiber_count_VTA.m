%% Compute the number of fibers lying in the VTA for each patient 
clear all; 

%% Load the fibers of interest (Petersen atlas, GPe->STN, STN->GPi - inidrect pathway
l_GPe_STN = load(['C:\Users\ICN\Documents\Try_without_Sdrive\structural_connectivity\' ...
    'Basal Ganglia Pathway Atlas (Petersen 2019) DISTAL\lh\gpe2stn_sm.mat']);
l_STN_GPi = load(['C:\Users\ICN\Documents\Try_without_Sdrive\structural_connectivity\' ...
    'Basal Ganglia Pathway Atlas (Petersen 2019) DISTAL\lh\stn2gpi_sm.mat']);
r_GPe_STN = load(['C:\Users\ICN\Documents\Try_without_Sdrive\structural_connectivity\' ...
    'Basal Ganglia Pathway Atlas (Petersen 2019) DISTAL\rh\gpe2stn_sm.mat']);
r_STN_GPi = load(['C:\Users\ICN\Documents\Try_without_Sdrive\structural_connectivity\' ...
    'Basal Ganglia Pathway Atlas (Petersen 2019) DISTAL\rh\stn2gpi_sm.mat']);
fibers_all = [[r_GPe_STN, l_GPe_STN]; [r_STN_GPi, l_STN_GPi]];


%% Loop over subjects and load VTAs
subjects = [2:4 6:25];
path = "C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\LeadDBSDataset\\derivatives\\leaddbs\\sub-%s" + ...
    "\\stimulations\\MNI152NLin2009bAsym\\gs_test\\sub-%s_sim-binary_model-simbio_hemi-%s.mat";

sides = ["R", "L"];
res = zeros(length(subjects), length(sides), length(fibers_all));
for i=1:length(subjects)
    sub_id = string(subjects(i));
    for j=1:length(sides)
        VTA_path = char(sprintf(path, sub_id, sub_id, sides(j))); 
        VTA = load(VTA_path);
        VTA = VTA.vatfv.vertices;
        max_VTA = [max(VTA(:, 1)), max(VTA(:, 2)), max(VTA(:, 3))];
        min_VTA = [min(VTA(:, 1)), min(VTA(:, 2)), min(VTA(:, 3))];
        
        % Loop over the fibers of interest
        for k=1:length(fibers_all)

            % Count the fibers that pass the VTA 
            fibers = fibers_all(k, j).fibers;
            fiber_IDs = [];
            % Loop over every fiber point 
            for l=1:length(fibers)
                fiber_dot = fibers(l, 1:3); 
                if (fiber_dot < max_VTA) & (fiber_dot > min_VTA)
                    fiber_IDs = cat(1, fiber_IDs, fibers(l, 4)); 
                end
            end
            % Count how many fibers (with at least 2 points)
            [count, elem] = groupcounts(fiber_IDs);
           %fibers_n = length(unique(fiber_IDs));
           fibers_n = sum(count == 3);
           res(i, j, k) = fibers_n;
        end
    end
end

% Save the fiber counts for later analysis
save("fiber_count_VTA.mat", "res")
