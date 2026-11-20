import os
import json

for model_name in ("model1", "model2", "model3"):
    history_path = os.path.join("Results", model_name, f"{model_name}_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            all_folds = json.load(f)
        print(f"[{model_name}] Key di history.json:", list(all_folds.keys()))
    else:
        print(f"[{model_name}] File tidak ditemukan: {history_path}")