import os

BASE_DIR = os.getcwd()+"/../../../"

FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")

# FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/merged/finger_embeddings.csv")
# AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/merged/quick_embeddings.csv")
# FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/merged/smile_embeddings.csv")

# FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/zero_padding/merged/zero_padding_finger_embeddings.csv")
# AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/zero_padding/merged/zero_padding_quick_embeddings.csv")
# FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/zero_padding/merged/zero_padding_smile_embeddings.csv")

# FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/average_padding/merged/average_padding_finger_embeddings.csv")
# AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/average_padding/merged/average_padding_quick_embeddings.csv")
# FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/average_padding/merged/average_padding_smile_embeddings.csv")


# Replace + zero padding
# FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/real_replace/finger_embeddings.csv")
# AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/real_replace/quick_embeddings.csv")
# FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/real_replace/smile_embeddings.csv")

# replace
# FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/cleaned_replace/finger_embeddings.csv")
# AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/cleaned_replace/quick_embeddings.csv")
# FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/synthetic_data/cleaned_replace/smile_embeddings.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal_ori', 'fox_model_best_auroc_baal_ori', 'facial_expression_smile_best_auroc_baal_ori'],
    # 1: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal'],
    # 2: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
    # 3: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']
}