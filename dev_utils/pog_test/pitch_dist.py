import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = '/scratch1/rsawahn/PeopleOnGrass/meta.csv'
pitch_column_name = 'gimbal_pitch(degrees)'

df = pd.read_csv(csv_file_path)
pitch_90 = df[(df[pitch_column_name] <= -85) & (df[pitch_column_name] >= -90)]
num_images_90_pitch = pitch_90.shape[0]
print(num_images_90_pitch)
pitch_data = df[pitch_column_name]


plt.figure(figsize=(10, 6))
plt.hist(pitch_data, bins=10, alpha=0.7, color='blue')
plt.title('Camera Pitch Distribution')
plt.xlabel('Pitch Value')
plt.ylabel('Frequency')
plt.grid(True)
# save the plot
plt.savefig('pitch_dist.png')
