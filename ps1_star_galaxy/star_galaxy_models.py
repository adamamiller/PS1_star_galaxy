import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier
import pickle


class WhiteFluxModel:
    """
    Generate a white flux model to separate stars and galaxies in PS1 data
    """

class RandomForestModel:
    """
    Random Forest Model to separate stars and galaxies in PS1 data
    
    
    """
    
    def __init__(self):
        pass
    
    def get_hst_train(self, file = "HST_COSMOS_features_adamamiller.fit", 
                      features = ['wwpsfChiSq', 'wwExtNSigma', 
                                  'wwpsfLikelihood', 'wwPSFKronRatio', 
                                  'wwPSFKronDist',  'wwPSFApRatio', 
                                  'wwmomentRH', 'wwmomentXX', 
                                  'wwmomentXY', 'wwmomentYY', 
                                  'wwKronRad']):
        """Get the training set for the RF model with HST-PS1 Xmatch sources 
        
        Parameters
        ----------
        file : str, file name (default: "HST_COSMOS_features_adamamiller.fit")
            Path to the fits file (or other astropy readable file) with the 
            features and labels for the HST training set
        
        features : list-like (default: ['wwpsfChiSq', 'wwExtNSigma', 
                                        'wwpsfLikelihood', 'wwPSFKronRatio', 
                                        'wwPSFKronDist',  'wwPSFApRatio', 
                                        'wwmomentRH', 'wwmomentXX', 
                                        'wwmomentXY', 'wwmomentYY', 
                                        'wwKronRad'])
            A list of features to use in training the RF model. Features must 
            correspond to columns in file.
        
        label : str (default: "MU_CLASS")
            The column name of the label data for the training set. label must 
            correspond to a single column in file.
        
        Attributes
        ----------
        hst_X_ : array-like
            The scikit-learn compatible feature array for the HST training set
        
        hst_y_ : array-like
            The scikit-learn compatible label array for the HST training set        
        """
        hst_df = Table.read("HST_COSMOS_features_adamamiller.fit").to_pandas()
        hst_det = np.where(hst_df.nDetections > 0)
        self.hst_X_ = np.array(hst_df[features].ix[hst_det])
        if label == "MU_CLASS":
            self.hst_y_ = np.array(hst_df["MU_CLASS"].ix[hst_det] - 1)
        else:
            self.hst_y_ = np.array(hst_df[label].ix[hst_det])
    
    def train_hst_rf(self, ntree=400, mtry=4, nodesize=2):
        """Train the RF on the HST training set
        
        Parameters
        ----------
        ntree : int (default: 400)
            The number of trees to include in the random forest model
        
        mtry : int (default: 4)
            The number of features searched at each node in the model
        
        nodesize : int (default: 2)
            The minimum acceptible number of samples for a terminal tree node
        
        Attributes
        ----------
        rf_clf_ : sklearn RandomForestClassifier object
            A sklearn RandomForestClassifier object trained on the HST training set 
        """
        
        if not hasattr(self, "hst_X_"):
            self.get_hst_train()
        
        rf_clf = RandomForestClassifier(n_estimators=ntree, 
                                        max_features=mtry,
                                        min_samples_leaf=nodesize,
                                        n_jobs=-1)
        rf_clf.fit(self.hst_X_, self.hst_y_)
        self.rf_clf_ = rf_clf
    
    def save_rf_as_pickle(self, pkl_file="final_hst_rf.pkl"):
        """Save the trained RF model as a pickle file
        
        Parameters
        ----------
        pkl_file : str, file path
            Full path to the pickle file that will store the HST-trained 
            random forest classification model
        """
        with open(pkl_file, "wb") as pf:
            pickle.dump( self.rf_clf_, pf)
    
    def read_rf_from_pickle(self, pkl_file="final_hst_rf.pkl"):
        """Load the trained RF model as a pickle file"""
        with open(pkl_file, "rb") as pf:
            self.rf_clf_ = pickle.load( pf )
    
    def classify_ps1_sources(self, ps1_fits_file,
                             features = ['wwpsfChiSq', 'wwExtNSigma', 
                                         'wwpsfLikelihood', 'wwPSFKronRatio', 
                                         'wwPSFKronDist',  'wwPSFApRatio', 
                                         'wwmomentRH', 'wwmomentXX', 
                                         'wwmomentXY', 'wwmomentYY', 
                                         'wwKronRad']):
        """Read in FITS from PS1 casjobs and classify sources"""
        ps1_df = Table.read(ps1_fits_file).to_pandas()
        ps1_X = np.array(ps1_df[features])
        rf_proba = self.rf_clf_.predict_proba(ps1_X)[:,1]
        df_out = ps1_df.copy()[['objid', 'raStack', 'decStack', 'qualityFlag']]
        df_out['rf_score'] = rf_proba
        out_file = ps1_fits_file.split("_features")[0] + "_classifications.h5"
        df_out.to_hdf(out_file, "class_table")
        