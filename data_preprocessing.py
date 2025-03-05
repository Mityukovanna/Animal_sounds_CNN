import os
import pandas as pd
from pydub import AudioSegment

# Define paths
data_folder = r'C:\Users\admin\.cache\kagglehub\datasets\ouaraskhelilrafik\tp-02-audio\versions\1\Data\Data'
wav_folder = r'C:\Users\admin\Desktop\python projects\neural_zoo\Wav_files'
csv_file = r'C:\Users\admin\Desktop\python projects\neural_zoo\labels.csv'  # Save outside the wav folder

def convert_ogg_to_wav(data_folder, wav_folder, csv_file):
    os.makedirs(wav_folder, exist_ok=True)  # Ensure output folder exists

    data = []  # Store filename-label pairs
    labels_set = set()  # Store unique labels for one-hot encoding

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".ogg"):
                ogg_path = os.path.join(root, file)
                
                # Extract the class label (e.g., "101-Dog" → "Dog")
                folder_name = os.path.basename(root)
                label = folder_name.split('-')[-1]  # Get only the animal name

                # Convert filename
                wav_filename = os.path.splitext(file)[0] + ".wav"
                wav_path = os.path.join(wav_folder, wav_filename)

                try:
                    # Convert OGG to WAV
                    audio = AudioSegment.from_ogg(ogg_path)
                    audio.export(wav_path, format="wav")

                    # Store filename and label
                    data.append([wav_filename, label])
                    labels_set.add(label)

                    print(f"✅ Converted {ogg_path} -> {wav_path} | Label: {label}")

                except Exception as e:
                    print(f"❌ Error converting {ogg_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(data, columns=["filename", "label"])

    # Perform one-hot encoding
    df_one_hot = pd.get_dummies(df['label'], prefix='class')

    # Concatenate filename column with one-hot encoded labels
    df_final = pd.concat([df['filename'], df_one_hot], axis=1)

    # Save to CSV
    df_final.to_csv(csv_file, index=False)

    print(f"\n✅ Labels saved in {csv_file} with one-hot encoding.")


# Run the function
convert_ogg_to_wav(data_folder, wav_folder, csv_file)
