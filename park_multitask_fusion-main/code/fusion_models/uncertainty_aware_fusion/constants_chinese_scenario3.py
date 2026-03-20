import os

# Scenario 3: Real Smile + Synthetic Speech (Bimodal)
BASE_DIR = os.getcwd()+"/../../../"

# Real smile data
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/chinese_smile_real.csv")

# No finger data - bimodal fusion only
FINGER_FEATURES_FILE = None

# Synthetic speech data  
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/scenario2_smile_finger/chinese_quick_synthetic.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, "models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

# Use only speech and smile models (bimodal)
MODEL_SUBSETS = {
    0: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}

