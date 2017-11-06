# ML-FO_Flow
ML Fuel Oil Flow

Machine Learning using AutoML, predicting Fuel Oil consumption for a cruise ship using AutoML methods. Using the Python Anaconda 3.6+ on Linux Ubuntu 16.04 LTS.

- The 'headers.dict.xlsx' file is for creating a dictionary of variable names
contained in the database. It needs to be manually edited in Excel.
- The 'var_names.py' is for creating the Python-dictionary.
- The 'train_model.py' contains the wrapping of the training functions. So far only the TPOTRegressor is included.

The training is called within a shell under a GNU SCREEN-session.

'python training_1-predictor.py | tee trained_models/training_log.log'
