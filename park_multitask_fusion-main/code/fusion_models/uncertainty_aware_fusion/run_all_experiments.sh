#!/bin/bash
# Run all remaining experiments

BASE_DIR="/localdisk2/pliu/park_multitask_fusion-main"
cd $BASE_DIR/code/fusion_models/uncertainty_aware_fusion

# Exp 2: Smile + Speech
echo "=== Running Experiment 2: Smile + Speech ==="
cat > constants.py << 'PYEOF'
import os
BASE_DIR = os.getcwd()+"/../../../"
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")
MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")
FACIAL_EXPRESSIONS = {'smile': True, 'surprise': False, 'disgust': False}
MODEL_SUBSETS = {0: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']}
PYEOF
conda run -n park python uncertainty_aware_fusion.py 2>&1 | tee $BASE_DIR/results/synthetic_validation/exp2_smile_speech.log | grep "auroc"

# Exp 3: Finger + Speech  
echo "=== Running Experiment 3: Finger + Speech ==="
cat > constants.py << 'PYEOF'
import os
BASE_DIR = os.getcwd()+"/../../../"
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")
MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")
FACIAL_EXPRESSIONS = {'smile': True, 'surprise': False, 'disgust': False}
MODEL_SUBSETS = {0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal']}
PYEOF
conda run -n park python uncertainty_aware_fusion.py 2>&1 | tee $BASE_DIR/results/synthetic_validation/exp3_finger_speech.log | grep "auroc"

echo "=== All experiments completed ==="
