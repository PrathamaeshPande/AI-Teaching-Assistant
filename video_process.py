import os
import subprocess

# Create output folder if it doesn't exist
os.makedirs("audios", exist_ok=True)

# Get all files from videos folder
files = os.listdir("videos")

for file in files:
    # Process only .mp4 files
    if file.endswith(".mp4"):
        input_path = os.path.join("videos", file)

        # Extract filename without extension (1.mp4 → 1)
        name = os.path.splitext(file)[0]

        output_path = os.path.join("audios", f"{name}.mp3")

        print(f"Converting: {file} → {name}.mp3")

        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            output_path
        ])