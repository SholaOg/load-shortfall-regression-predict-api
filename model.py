"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    
    # create new features
    #Converting the object in the "time" to datetime 
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'], format='%Y-%m-%d %H:%M:%S')

    #Creting new features and re-ordering the df
    feature_vector_df['year'] = pd.DatetimeIndex(feature_vector_df['time']).year
    feature_vector_df['month'] = pd.DatetimeIndex(feature_vector_df['time']).month
    feature_vector_df['day'] = pd.DatetimeIndex(feature_vector_df['time']).day
    feature_vector_df['hour'] = pd.DatetimeIndex(feature_vector_df['time']).hour
    
    #Reordering df columns
    column_titles = ['time', 'year', 'month', 'day', 'hour', 'Madrid_wind_speed', 
                 'Valencia_wind_deg', 'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity', 
                 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all', 
                 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all', 
                 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_pressure', 'Seville_rain_1h', 
                 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 
                 'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure', 
                 'Seville_weather_id', 'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 
                 'Valencia_temp', 'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min', 
                 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 
                 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min', 
                 'load_shortfall_3h']
    df=feature_vector_df.reindex(columns=column_titles)
    
    #Fill Valencia_pressure with the mean of each time of the day and day of the month.
    for label, row in df.iterrows():
        if pd.isnull(row['Valencia_pressure']):
            y = row['year']
            m = row['month']
            d = row['day']
            h = row['hour']

            df.loc[label, 'Valencia_pressure'] = round((df[(df['year'] == y) & (df['month'] == m)]['Valencia_pressure']).mean(),1) 
    
    #df_dummy =df_dummy.reindex(columns=column_titles)
    df_dummies = pd.get_dummies(df)

    # Again we make sure that all the column names have underscores instead of whitespaces
    df_dummies.columns = [col.replace(" ","_") for col in df_dummies.columns] 
    
    #Reindexing the new DF such the load_shortfall_3h becomes the last column.
    column_titles = [col for col in df_dummies.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
    df_dummies=df_dummies.reindex(columns=column_titles)
    
    from scipy.stats import pearsonr
    # Build a dictionary of correlation coefficients and p-values
    dict_cp = {}

    column_titles = [col for col in corrs.index if col!= 'load_shortfall_3h']
    for col in column_titles:
        p_val = round(pearsonr(df_dummies[col], df_dummies['load_shortfall_3h'])[1],6)
        dict_cp[col] = {'Correlation_Coefficient':corrs[col],
                        'P_Value':p_val}

    df_cp = pd.DataFrame(dict_cp).T
    df_cp_sorted = df_cp.sort_values('P_Value')
    df_cp_sorted[df_cp_sorted['P_Value']<0.1]
    
    #Building model with feature having p valve < 0.05
    # The dependent variable remains the same:
    y_data = df_dummies['load_shortfall_3h']  # y_name = 'load_shortfall_3h'

    # Model building - Independent Variable (IV) DataFrame
    X_names = list(df_cp[df_cp['P_Value'] < 0.05].index)
    X_data = df_dummies[X_names]
    
    X_remove = ['Valencia_temp_min', 'Valencia_temp_max',
            'Seville_temp_min', 'Seville_temp_max',
            'Bilbao_temp_min', 'Bilbao_temp_max',
            'Barcelona_temp_min', 'Barcelona_temp_max',
            'Madrid_temp_min', 'Madrid_temp_max', 
            'Seville_humidity', 'Madrid_humidity',
            'Valencia_snow_3h', 'Sevill_rain_3h']
             
    X_corr_names = [col for col in X_names if col not in X_remove]
    predict_vector = X_data[X_corr_names]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
