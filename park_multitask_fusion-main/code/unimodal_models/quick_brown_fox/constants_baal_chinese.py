import os

# 中文数据测试配置
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
BASE_PATH = BASE_DIR

# 中文数据路径 - processed目录下的三个模态数据
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/facial_dataset.csv")
WAVLM_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/wavlm_fox_features.csv")

# 兼容旧的quick_brown_fox相关变量名
CLASSICAL_FEATURES_FILE = WAVLM_FEATURES_FILE  # 不使用，保留兼容性
IMAGEBIND_FEATURES_FILE = WAVLM_FEATURES_FILE  # 不使用，保留兼容性
WAV2VEC_FEATURES_FILE = WAVLM_FEATURES_FILE    # 不使用，保留兼容性

# 使用训练好的模型
MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = os.path.join(BASE_DIR, f"models/fox_model_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model.pth")
SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model_config.json")