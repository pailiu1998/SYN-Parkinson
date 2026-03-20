import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景3: 合成Finger + 真实Quick + 合成Smile
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_3_real_quick/features_demography_diagnosis.csv")  # 合成
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_3_real_quick/wavlm_fox_features.csv")  # 真实
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/mixed_scenario_3_real_quick/facial_dataset.csv")  # 合成

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}

