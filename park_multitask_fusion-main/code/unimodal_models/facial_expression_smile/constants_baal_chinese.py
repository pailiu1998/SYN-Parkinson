import os

# 中文数据测试配置
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
BASE_PATH = BASE_DIR

# 面部表情配置
FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

# 中文Smile数据路径
FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/facial_dataset.csv")

# 使用训练好的模型
MODEL_TAG = "best_auroc_baal"

MODEL_BASE_PATH = os.path.join(BASE_DIR, f"models/facial_expression_smile_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model_config.json")
SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler/scaler.pth")

