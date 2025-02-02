import subprocess

result = subprocess.run(
    ["python", "Face_Recognition.py"],  # Replace with the actual script filename
    text=True,
    capture_output=True
)

detected_name = result.stdout.strip()

if detected_name:
    print(f"Detected person: {detected_name}")
else:
    print("No face detected.")
