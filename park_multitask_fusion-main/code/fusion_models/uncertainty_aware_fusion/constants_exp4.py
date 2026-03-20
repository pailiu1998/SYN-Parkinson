import os

BASE_DIR = os.getcwd()+"/../../../"

# Experiment 4: Real Smile + Real Finger + Synthetic Speech
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/park_train_synthetic_merged_filtered/quick_embeddings.csv")  # SYNTHETIC

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}
