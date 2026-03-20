import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景1: 真实Finger + 合成Quick + 合成Smile
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_1_real_finger/features_demography_diagnosis.csv")  # 真实
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_1_real_finger/wavlm_fox_features.csv")  # 合成
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/mixed_scenario_1_real_finger/facial_dataset.csv")  # 合成

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}

