import os

# current dir: /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping/
BASE_DIR = os.getcwd()+"/../../../"
# base dir: /localdisk2/pliu/park_multitask_fusion-main/

# Use Dragon-PD dataset
FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/dragon_pd_features.csv")

MODEL_TAG = "dragon_pd_test"

MODEL_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}","predictive_model/model.pth")
SCALER_PATH = os.path.join(BASE_DIR, f"models/finger_model_{MODEL_TAG}","scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, f"models/finger_model_{MODEL_TAG}","predictive_model/model_config.json")
MODEL_BASE_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}")


