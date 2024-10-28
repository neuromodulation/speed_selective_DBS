%% Quick analysis of the behavioral data of the vigor stim dataset
% 1. Show the averaged velocity values in both conditions (slow/fast) 
% 2. Measure the task performance 
% -> Inclusion criteria for the participants
% 1. Correct amount of movements are stimulated (between 25-40 %)
% 2. Correct movements are stimulated (sensitivity + specificity)
% 3. Time of stimulation is close to true peak 

% Load the data from one participant
[filename,path] = uigetfile('..\..\.. \Data\');
load(strcat(path,filename));
data = struct.data; 
options = struct.options;

% Set the parameters 
n_trials = 96;
n_blocks = 2; % number of blocks per condition 
n_cond = 2; % number of conditions
% Change the order in which the blocks are processed such that the order is
% always slow - first (cond=1)
if options.cond
    stim_blocks = [1:2; 3:4];
else
    stim_blocks = [3:4; 1:2];
end

% Initialize the arrays
peaks_all = []; % array to store the peak velocities from every movement
diffs_stim_peak = []; % array to store the time difference between peak and stimulation
perf = []; % array to store the percentage of stimulated movements/sensitivity/specificity

% Loop over stimulation conditions(slow-fast)
for i_cond=1:n_cond
    
    stims = []; % array to store whether a trial was stimulated or not
    true_stims = []; % array to store whether a trial should be stimulated or not
    peaks = []; % array to store the peak
    
    % Loop over the blocks (stimulation-recovery)
    for i_block=1:n_blocks
            
        % Loop over every movement
        for i_trial=1:n_trials
            % Get the data from one trial
            mask = data(:,8) == stim_blocks(i_cond,i_block) & data(:,9) == i_trial;
            data_trial = data(mask,:); 
            % Get the peak and peak index
            [peak, ind_peak] = max(data_trial(:,4));
            peaks = cat(1,peaks, peak);
            
            if i_block == 1
                % Check whether the movement should have been stimulated 
                if length(peaks)> 2 && (all(peaks(end-2:end-1) > peaks(end)) && i_cond == 1 ||...
                    all(peaks(end-2:end-1)< peaks(end)) && i_cond == 2)
                    true_stim = 1;
                else
                    true_stim = 0;
                end
                % Check whether movement was stimulated
                inds_stim = find(data_trial(:,11) == 1);
                if inds_stim
                    stim = 1;
                    % Get the index of the stimulation 
                    ind_stim = inds_stim(1); 
                    % Save the time difference between peak and stimulation 
                    diff_stim_peak = data_trial(ind_stim,3) - data_trial(ind_peak,3);
                    diffs_stim_peak = cat(1, diffs_stim_peak, diff_stim_peak);
                else
                    stim = 0;
                end
                % Save whether the movement was stimulated and should have been
                % stimulated
                stims = cat(1,stims, stim);
                true_stims = cat(1,true_stims, true_stim);
            end
        end
        
        if i_block == 1
            % Calculate the number of stimulated movements as well as sensitivity
            % and specificity
            perc_stim = sum(stims)/length(stims);
            sens = sum(stims == 1 & true_stims == 1)/sum(true_stims == 1);
            spec = sum(stims == 0 & true_stims == 0)/sum(true_stims == 0);
            perf = cat(1,perf,[perc_stim, sens,spec]*100);
        end
    end
    % If needed, fill the outliers of the data
    peaks = filloutliers(peaks,"linear");
    
    % Save the peak velocities for one condition 
    peaks_all = cat(2,peaks_all, peaks);
end

% Plot the results 
figure;

% Plot the peak velocity values over trials for each condition 
subplot(3,1,1);
% Normalize the peak velocity values by substracting the mean velocity of
% the movements 5-15
peaks_all = peaks_all% - mean(peaks_all(5:15,:),1);
% Average the values over 8 consecutive movements
peaks_mean = peaks_all%squeeze(mean(reshape(peaks_all.',2,8,[]),2));
% Plot
x = linspace(1,length(peaks_all),length(peaks_mean));
plot(x, peaks_mean, "LineWidth", 2);
ylabel("Normalized peak velocity (pixel/sec)"); 
xlabel("Movement number");
xline(x(length(x)/2), "LineWidth", 2); % indicate the start of the recovery block
legend(["Slow", "Fast", "Start Recovery"]);

% Plot the time between true peak and stimulation
subplot(3,1,2);
histogram(diffs_stim_peak,20);
xlabel("Time from peak to start of stimulation (sec)");

% Plot the performance values
subplot(3,1,3);
b = bar(perf);
ylabel("%");
set(b, {'DisplayName'}, {'Total %','Sensitivity','Specificity'}')
legend();
set(gca, 'XTick', 1:2, 'XTickLabels', {'Slow','Fast'})
% Add the values 
width = b.BarWidth;
for i=1:length(perf(:, 1))
    row = perf(i, :);
    % 0.5 is approximate net width of white spacings per group
    offset = ((width + 0.5) / length(row)) / 2;
    x = linspace(i-offset, i+offset, length(row));
    text(x,row,num2str(row'),'vert','bottom','horiz','center');
end
sgtitle(filename);

