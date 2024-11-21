# Save with average ref added

import sys
from mne_bids import read_raw_bids, find_matching_paths
sys.path.insert(1, "../../../Code")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects="EL012",
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMedOff01", f"EcogLfpMedOff01",
                                              f"LfpMedOff02", f"EcogLfpMedOff02", f"LfpMedOffDys01"])
raw = read_raw_bids(bids_path=bids_path[0])
raw.load_data()

# Add re-references ECoG channels
ecog_names = ["ECOG_R_1_CAR_12345", "ECOG_R_2_CAR_12345", "ECOG_R_3_CAR_12345", "ECOG_R_4_CAR_12345", "ECOG_R_5_CAR_12345"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT", "ECOG_R_04_SMC_AT", "ECOG_R_05_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")

# Add the bipolar LFP channel
lfp_names = ["LFP_R_1_BIP_234_1", "LFP_R_2_BIP_234_8"]
og_chan_names = ["LFP_R_01_STN_MT", "LFP_R_08_STN_MT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(["LFP_R_02_STN_MT", "LFP_R_03_STN_MT", "LFP_R_04_STN_MT"]).sum(axis=0) - raw.get_data(chan)
    u.add_new_channel(raw, new_ch, lfp_names[i], type="dbs")

# Save
save_path = f"EL012_ECoG_CAR_LFP_BIP.fif"
raw.save(save_path, overwrite=True)

# Save this with all other LFP/ECoG channels deleted
raw = raw.pick(raw.info["ch_names"][32:])
save_path = f"EL012_ECoG_CAR_LFP_BIP_small.fif"
raw.save(save_path, overwrite=True)

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects="EL012",
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMedOff01", f"EcogLfpMedOff01",
                                              f"LfpMedOff02", f"EcogLfpMedOff02", f"LfpMedOffDys01"])
raw = read_raw_bids(bids_path=bids_path[0])
raw.load_data()

# Add re-references ECoG channels
ecog_names = ["ECOG_R_1_CAR_123", "ECOG_R_2_CAR_123", "ECOG_R_3_CAR_123"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")

# Add the bipolar LFP channel
lfp_names = ["LFP_R_1_BIP_234_1", "LFP_R_2_BIP_234_8"]
og_chan_names = ["LFP_R_01_STN_MT", "LFP_R_08_STN_MT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(["LFP_R_02_STN_MT", "LFP_R_03_STN_MT", "LFP_R_04_STN_MT"]).sum(axis=0) - raw.get_data(chan)
    u.add_new_channel(raw, new_ch, lfp_names[i], type="dbs")

# Save
save_path = f"EL012_ECoG_123_CAR_LFP_BIP.fif"
raw.save(save_path, overwrite=True)

# Save this with all other LFP/ECoG channels deleted
raw = raw.pick(raw.info["ch_names"][32:])
save_path = f"EL012_ECoG_123_CAR_LFP_BIP_small.fif"
raw.save(save_path, overwrite=True)


