import os

BASE_DIR = os.getcwd()+"/../../../"

# Diffusion: Real Quick + Real Finger + Synthetic Smile
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/converted/diffusion_2real_1synth/3_quick_finger_REAL__smile_SYNTH/features_demography_diagnosis.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/converted/diffusion_2real_1synth/3_quick_finger_REAL__smile_SYNTH/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/converted/diffusion_2real_1synth/3_quick_finger_REAL__smile_SYNTH/wavlm_fox_features.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}
