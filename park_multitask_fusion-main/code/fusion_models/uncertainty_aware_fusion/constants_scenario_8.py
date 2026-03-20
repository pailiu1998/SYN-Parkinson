import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景8: 真实 Quick，合成 Smile + Finger (single_to_dual/quick_to_smile_finger)
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/single_to_dual/quick_to_smile_finger/replaced_data/features_demography_diagnosis.csv")  # 合成
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/single_to_dual/quick_to_smile_finger/replaced_data/wavlm_fox_features.csv")  # 真实
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/single_to_dual/quick_to_smile_finger/replaced_data/facial_dataset.csv")  # 合成

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


