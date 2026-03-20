import os

# Scenario 1: Real Smile + Synthetic Finger + Synthetic Speech
BASE_DIR = os.getcwd()+"/../../../"

# Real smile data
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv")

# Synthetic finger data (merged left and right hands from scenario1)
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario1_smile_only/chinese_finger_synthetic.csv")

# Synthetic speech data
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario1_smile_only/chinese_quick_synthetic.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, "models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

# Use all three trained unimodal models
MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}

