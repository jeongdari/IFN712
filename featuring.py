import pandas as pd
import numpy as np

# Load your dataset
file_path = 'Merged_GNSS_and_SNR_Data.csv'
data = pd.read_csv(file_path)

# Convert 'TIME' column to datetime if it exists
if 'TIME' in data.columns:
    data['TIME'] = pd.to_datetime(data['TIME'], dayfirst = True)

# Filter rows where Pseudorange Residual (m) = 0
zero_residual_data = data[data['Pseudorange Residual (m)'] == 0]

# Statistical summary of features when Pseudorange Residual = 0
mean_snr_when_zero = zero_residual_data['SNR(dBHz)'].mean()
median_snr_when_zero = zero_residual_data['SNR(dBHz)'].median()
std_snr_when_zero = zero_residual_data['SNR(dBHz)'].std()

# Create time-related features (Hour, Day, Weekday)
if 'TIME' in data.columns:
    data['hour'] = data['TIME'].dt.hour
    data['day'] = data['TIME'].dt.day
    data['weekday'] = data['TIME'].dt.weekday

# Rolling statistics for SNR
if 'SNR(dBHz)' in data.columns:
    data['SNR_mean_5'] = data['SNR(dBHz)'].rolling(window=5).mean()
    data['SNR_std_5'] = data['SNR(dBHz)'].rolling(window=5).std()

# Lag features for pseudorange residuals and SNR
if 'Pseudorange Residual (m)' in data.columns and 'SNR(dBHz)' in data.columns:
    data['Pseudorange_Residual_lag1'] = data['Pseudorange Residual (m)'].shift(1)
    data['SNR_lag1'] = data['SNR(dBHz)'].shift(1)
    
    # Difference features
    data['Pseudorange_Residual_diff'] = data['Pseudorange Residual (m)'].diff()
    data['SNR_diff'] = data['SNR(dBHz)'].diff()

# Feature: Interaction between SNR and Elevation
if 'SNR(dBHz)' in data.columns and 'Elevation' in data.columns:
    mean_elevation_when_zero = zero_residual_data['Elevation'].mean()
    std_elevation_when_zero = zero_residual_data['Elevation'].std()
    
    # Create interaction feature
    data['SNR_Elevation_interaction'] = data['SNR(dBHz)'] * data['Elevation']
    
    # Feature: Is the current elevation close to the mean elevation when residual is zero?
    data['Elevation_close_to_zero_residual'] = np.abs(data['Elevation'] - mean_elevation_when_zero) < std_elevation_when_zero

# Fourier Transform on pseudorange residual
if 'Pseudorange Residual (m)' in data.columns:
    fft_features = np.fft.fft(data['Pseudorange Residual (m)'].fillna(0).values)
    data['fft_real'] = np.real(fft_features)
    data['fft_imag'] = np.imag(fft_features)

# Feature: Is the current SNR close to the mean SNR when residual is zero?
data['SNR_close_to_zero_residual'] = np.abs(data['SNR(dBHz)'] - mean_snr_when_zero) < std_snr_when_zero

# Binary feature: Is the SNR higher than the average SNR when residual is zero?
data['SNR_higher_than_zero_residual_mean'] = data['SNR(dBHz)'] > mean_snr_when_zero

# Dropping rows with NaN values created by rolling or shifting
data = data.dropna()

# View the first few rows of the dataset with new features
print(data.head())

# Save the new dataset with engineered features
output_file = 'Merged_HKSL_with_all_features.csv'
data.to_csv(output_file, index=False)
print(f"Feature-engineered dataset saved to: {output_file}")
