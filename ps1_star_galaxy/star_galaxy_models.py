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
        """
        hst_df = Table.read("HST_COSMOS_features_adamamiller.fit").to_pandas()
        hst_det = np.where(hst_df.nDetections > 0)
        self.hst_X = np.array(hst_df[features].ix[hst_det])
        self.hst_y = np.array(hst_df["MU_CLASS"].ix[hst_det] - 1)
    
    def train_hst_rf(self, ntree=400, mtry=4, nodesize=2):
        """Train the RF on the HST training set
        """
        
        if not hasattr(self, "hst_X"):
            self.get_hst_train()
        
        rf_clf = RandomForestClassifier(n_estimators=ntree, 
                                        max_features=mtry,
                                        min_samples_leaf=nodesize,
                                        n_jobs=-1)
        rf_clf.fit(self.hst_X, self.hst_y)
        self.rf_clf_ = rf_clf
    
    def save_rf_as_pickle(self, pkl_file="final_hst_rf.pkl"):
        """Save the trained RF model as a pickle file"""
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
        