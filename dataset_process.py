import pickle
import numpy as np
import os

def inject_aerospace_degradation(iq_data, max_doppler_rate=0.05):
    N, channels, L = iq_data.shape
    n_axis = np.arange(L)
    f_d = max_doppler_rate * (n_axis / L) 
    phase_integral = 2 * np.pi * np.cumsum(f_d)
    rotation_operator = np.exp(1j * phase_integral)
    complex_signal = iq_data[:, 0, :] + 1j * iq_data[:, 1, :]
    corrupted_signal = complex_signal * rotation_operator
    new_iq_data = np.zeros_like(iq_data)
    new_iq_data[:, 0, :] = np.real(corrupted_signal)
    new_iq_data[:, 1, :] = np.imag(corrupted_signal)
    return new_iq_data

def process_pkl_dataset(input_pkl_path, output_pkl_path, max_doppler_rate=0.05):
    if not os.path.exists(input_pkl_path):
        print(f"Error: Input file not found {input_pkl_path}")
        return

    print(f"Loading original dataset: {input_pkl_path}")
    with open(input_pkl_path, 'rb') as f:
        Xd = pickle.load(f, encoding='bytes') 
    
    print(f"Successfully loaded, contains {len(Xd)} dataset slices (keys).")
    
    Xd_corrupted = {}
    
    total_keys = len(Xd)
    current_count = 0

    print(f"Starting Doppler shift injection (max_doppler_rate={max_doppler_rate})...")
    for key, iq_data in Xd.items():
        current_count += 1
        
        mod_type = key[0].decode() if isinstance(key[0], bytes) else key[0]
        snr_val = key[1]
        
        if len(iq_data.shape) != 3 or iq_data.shape[1] != 2 or iq_data.shape[2] != 128:
            print(f"  [Warning] Abnormal shape {iq_data.shape} found for {mod_type} {snr_val}dB, skipping.")
            Xd_corrupted[key] = iq_data
            continue

        corrupted_data = inject_aerospace_degradation(iq_data, max_doppler_rate)
        
        Xd_corrupted[key] = corrupted_data.astype(np.float32)
        
        if current_count % 22 == 0:
            print(f"  Progress: {current_count}/{total_keys} completed. Currently processing: {mod_type} @ {snr_val}dB")

    print(f"Data processing completed. Saving as new pkl file: {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(Xd_corrupted, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Save successful! Structure fully aligned with original dataset.")

if __name__ == "__main__":
    INPUT_FILE = r"data\radioml\RML2016.10a_dict.pkl"
    OUTPUT_FILE = r"data\radioml\RML2016.10a_Aerospace_corrupted.pkl"
    
    process_pkl_dataset(INPUT_FILE, OUTPUT_FILE)
