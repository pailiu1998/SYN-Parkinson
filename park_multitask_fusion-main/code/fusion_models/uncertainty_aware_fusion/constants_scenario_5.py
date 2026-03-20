import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景5: 真实 Smile + Finger，合成 Quick
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_5_real_smile_finger/features_demography_diagnosis_Nov22_2023.csv")  # 真实
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_5_real_smile_finger/wavlm_fox_features.csv")  # 合成
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/mixed_scenario_5_real_smile_finger/facial_dataset.csv")  # 真实

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
    # 1: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal'],
    # 2: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
    # 3: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']
}


