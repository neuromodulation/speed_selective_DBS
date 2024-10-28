% Check stimulation contact and amplitudes (based on what was saved)

res = [];
for i=6:25
    if i < 10
        folder = '..\..\Data\Off\raw_data\0' + string(i) + '\';
    else
        folder = '..\..\Data\Off\raw_data\' + string(i) + '\';
    end
    listdir = dir(folder);
    for j=1:length(listdir)
        if endsWith(listdir(j).name,".mat")
            file = listdir(j).name;
            break
        end
    end
    disp(file)
    data = load(strcat(listdir(1).folder,'\', file)).struct;
    contact_L = data.options.contacts_L;
    contact_R = data.options.contacts_R;
    amp_R = data.options.stim_amp_R; 
    amp_L = data.options.stim_amp_L;
    res = cat(1, res, [contact_R, contact_L, amp_R, amp_L]);
end
% Save as excel table
disp("Test")
xlswrite('stim_contacts_amps.xlsx',res)
