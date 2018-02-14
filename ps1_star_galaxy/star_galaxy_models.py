import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle


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
        
    def create_training_set(self, ps1_fits_file, features = None, 
                            training_set = False, target_name = 'class', 
                            random_state = None):
        
        """
        Generate training set attribute from fits file
        
        Starting from a fits file that contains output from the PanSTARRS1 
        DR1 database, specifically the StackObjectAttributes table, this 
        function reads the file and generates features for use in scikit-learn
        machine learning models. The features are stored as a pandas DataFrame
        attribute of the class.
        
        Note - the following columns from the PS1 StackObjectAttributes are 
        assumed to be in ps1_fits_file (if they are missing the code will 
        break):
        
            gpsfMajorFWHM, gpsfMinorFWHM,
            gpsfCore, gpsfLikelihood, gpsfChiSq, gExtNSigma,
            gmomentXX, gmomentXY, gmomentYY, gmomentR1, gmomentRH, gKronRad,
            gPSFFlux, gKronFlux
        
        as well as similar entries in the same row for rizy filters.
                            
        Furthermore - for some sources PS1 was unable to measure some of 
        these features. That is typically indicated by a value of -999. This 
        code handles missing data by converting the respective features to a 
        value of 1e6. Note - sources with psfMajorFWHM = 0 do not necessarily
        indicate missing data, this needs to be investigated further.
                            
        Parameters
        ----------
        ps1_fits_file : file
            Fits file with PanSTARRS1 photometric properties stored as the 
            individual rows.
        
        features : list
            List of features to be included in the RF model
        
        training_set : bool (default = False)
            Boolean variable to indicate whether the data set being read 
            includes target labels for the individual sources for training
            the machine learning model.
        
        target_name : str (default = 'class')
            Name of the label to be fit by the RF model
        
        random_state : int, RandomState instance or None, optional (def.=None)
            - If int, random_state is the seed used by the random number 
              generator; 
            - If RandomState instance, random_state is the random number 
              generator; 
            - If None, the random number generator is the RandomState 
              instance used by np.random.
        
        Returns
        -------
        self : object
            Returns self.
        
        """
        
        raw_ps1_table = Table.read(ps1_fits_file)
        raw_ps1_df = raw_ps1_table.to_pandas()
        
        if training_set:
            if target_name not in list(raw_ps1_df.columns):
                raise ValueError("target_name must exist in ps1_fits_file" +
                                 " to build model on training set")
            else:
                self.y_df_ = raw_ps1_df[target_name]
                
        no_norm = ['gpsfCore', 'gpsfLikelihood', 'gpsfChiSq', 'gExtNSigma', 
                   'rpsfCore', 'rpsfLikelihood', 'rpsfChiSq', 'rExtNSigma', 
                   'ipsfCore', 'ipsfLikelihood', 'ipsfChiSq', 'iExtNSigma', 
                   'zpsfCore', 'zpsfLikelihood', 'zpsfChiSq', 'zExtNSigma', 
                   'ypsfCore', 'ypsfLikelihood', 'ypsfChiSq', 'yExtNSigma']
        self.X_df_ = raw_ps1_df[no_norm]
        
        normed_feat_list = ['momentXX', 'momentXY', 'momentYY', 
                            'momentR1', 'KronRad', 'momentRH']
        
        psf_flux_w = np.zeros(len(raw_ps1_df))
        kron_flux_w = np.zeros(len(raw_ps1_df))
        
        for filt in ['g', 'r', 'i', 'z', 'y']:
            psf_list = [filt + 'psfMajorFWHM', filt + 'psfMinorFWHM']
            filt_seeing = np.mean(raw_ps1_df[psf_list], axis = 1)
            
            feat_norm_terms = np.array([filt_seeing**2, filt_seeing**2, 
                                        filt_seeing**2, filt_seeing, 
                                        filt_seeing, filt_seeing**0.5]).T
            no_psf_model = np.where(filt_seeing <= 0)
            feat_norm_terms[no_psf_model] = -0.0    
                        
            ff_list = [filt + feat for feat in normed_feat_list]
            ff_norm_list = [filt + feat + "norm" for feat in normed_feat_list]

            ff_kwargs = {x : raw_ps1_df[y]/feat_norm_terms[:,z] 
                for x, y, z, in zip(ff_norm_list, ff_list, range(6))}
            
            self.X_df_ = self.X_df_.assign(**ff_kwargs)
        
            psf_det = np.array(raw_ps1_df[filt + 'PSFFlux'] > 0).astype(int)
            kron_det = np.array(raw_ps1_df[filt + 'KronFlux'] > 0).astype(int)
            
            psf_det_flux = psf_det*raw_ps1_df[filt + 'PSFFlux'].values
            kron_det_flux = kron_det*raw_ps1_df[filt + 'KronFlux'].values
                        
            filt_flux_ratio = np.divide(psf_det_flux, kron_det_flux, 
                                        out = np.zeros_like(psf_det_flux), 
                                        where = kron_det_flux != 0)
            f_kwargs = {filt + 'psfKronRatio' : filt_flux_ratio}
            self.X_df_ = self.X_df_.assign(**f_kwargs)
            
            psf_flux_w += psf_det_flux
            kron_flux_w += kron_det_flux
            
        w_flux_ratio = np.divide(psf_flux_w, kron_flux_w, 
                                 out = np.zeros_like(psf_flux_w), 
                                 where = kron_flux_w != 0)
        w_kwargs = {'wpsfKronRatio' : w_flux_ratio}
        self.X_df_ = self.X_df_.assign(**w_kwargs)
        self.X_df_.replace([-np.inf, np.inf], 1e6, inplace = True)
    
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
        hst_det = np.where(hst_tbl.nDetections > 0)
        self.hst_X = np.array(hst_tbl[features].ix[hst_det])
        self.hst_y = np.array(hst_tbl["MU_CLASS"].ix[hst_det] - 1)
    
    def train_hst_rf(self, ntree=400, mtry=4, nodesize=2):
        """Train the RF on the HST training set
        """
        
        if not hasattr(self, hst_X):
            self.get_hst_train()
        
        rf_clf = RandomForestClassifier(n_estimators=ntree, 
                                        max_features=mtry,
                                        min_samples_leaf=nodesize,
                                        n_jobs=-1)
        rf_clf.fit(self.hst_X, self.hst_Y)
        self.rf_clf_ = rf_clf
    
    def save_rf_as_pickle(self, pkl_file="final_hst_rf.pkl"):
        """Save the trained RF model as a pickle file"""
        with open(pkl_file, "wb") as pf:
            pickle.dump( self.rf_clf_, pf)
    
    def read_rf_from_pickle(self, pkl_file="final_hst_rf.pkl"):
        """Load the trained RF model as a pickle file"""
        with open(pkl_file, "rb") as pf:
            self.rf_clf_ = pickle.load( pf )
    