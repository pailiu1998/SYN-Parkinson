import os

# 中文数据测试配置
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"

# 中文Finger数据路径
FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/features_demography_diagnosis_Nov22_2023.csv")

# 使用训练好的模型
MODEL_TAG = "both_hand_fusion_baal"

MODEL_BASE_PATH = os.path.join(BASE_DIR, f"models/finger_model_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model.pth")
SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model_config.json")

