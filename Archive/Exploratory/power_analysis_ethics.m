% settings
alpha_level = 0.001;   % alpha level
power_level = 0.9;     % desired power level
p_heads_h1  = 0.55;    % chance to land on head under H1
n_start     = 20;      % sample size to start
n_sim       = 1000;    % number of experiments

% run simulations for each sample size until power reaches desired level
power      = 0;                  % initialize the variable
power_at_n = zeros(1,n_start);   % initialize
n_sample   = n_start;            % sample size
while power < power_level        % continue increasing sample-size until power reach desired level
    n_sample = n_sample + 1;     % sample size in current iteration
    disp(n_sample);
    n_heads = binornd(n_sample, p_heads_h1, [1,n_sim]);   
    p_h0  = 1-binocdf(n_heads-1, n_sample, 0.5);  
    power = sum(p_h0 < alpha_level)/n_sim;  % calculate power for the current sample size
    power_at_n(n_sample) = power;
end

% plot the result
plot(1:n_sample, power_at_n, 'ko'); hold on
plot([0 n_sample],[power_level power_level],'r');
xlim([n_start+1 n_sample]); ylim([0 1])
xlabel('Sample size'); ylabel('Power')
title(['Required sample size is' num2str(n_sample)])