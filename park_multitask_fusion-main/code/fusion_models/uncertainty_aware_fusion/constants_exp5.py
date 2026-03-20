import os

BASE_DIR = os.getcwd()+"/../../../"

# Experiment 5: Real Smile + Synthetic Finger + Real Speech
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/park_train_synthetic_merged_filtered/finger_embeddings.csv")  # SYNTHETIC
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}
