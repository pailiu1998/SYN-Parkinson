# Multi-task Fusion for Parkinson's Screening

This is currently a private repository, for internal collaboration and discussion.

To begin with, we will maintain the following directory structure:

**ROOT/**

  **code/**
  
    unimodal_models/
      unimodal_finger.py
      ...
        
    fusion_models/
      ...
  
  **data/**
  
    finger_tapping/
        
      features with label (csv file)
      
  **models/**
  
    finger_model_{tag}/
    
      predictive_model/
        model_config.json (dictionary; hyper-parameters passed as command line arguments)
        model.pth (torch model)
          
      scaler/
        scaler.pth (can be empty if hyper-parameter "use_feature_scaler" is set to :"no")
