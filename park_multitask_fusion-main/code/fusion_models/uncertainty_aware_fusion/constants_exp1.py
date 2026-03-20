import os

BASE_DIR = os.getcwd()+"/../../../"

# Use only Finger + Smile for Experiment 1
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")  # Not used in exp1

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

# Experiment 1: Smile + Finger only
MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],  # Default to exp1
    1: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal'],
    2: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
    3: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']
}
