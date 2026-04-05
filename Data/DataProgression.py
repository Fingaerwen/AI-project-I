import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
YOLO_LABEL_DIR = os.path.join(BASE_DIR, "Labels", "yolo_labels", "obj_Train_data", "Training")

rows = []

for file_name in os.listdir(IMAGE_DIR):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_DIR, file_name)
        label_file_name = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(YOLO_LABEL_DIR, label_file_name)

        if os.path.exists(label_path):
            rows.append({
                "input": image_path,
                "output": label_path
            })

data_df = pd.DataFrame(rows, columns=["input", "output"])

print("Total samples found:", len(data_df))
print(data_df.head())

if len(data_df) == 0:
    raise ValueError("No matching image/label pairs were found. Check filenames and paths.")

train_df, val_df = train_test_split(
    data_df,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE_DIR, "val.csv"), index=False)

print("CSV files generated successfully.")