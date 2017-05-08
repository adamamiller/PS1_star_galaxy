import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits


class RandomForestModel:
    """
    Random Forest Model to separate stars and galaxies in PS1 data
    
    
    """
    
    def __init__(self):
        pass
        
    def create_training_set(self, ps1_fits_file, features = None, 
                            target_name = 'class', random_state = None):
        """
        Generate training set attribute from fits file
        
        Parameters
        ----------
        ps1_fits_file : file
            Fits file with PanSTARRS1 photometric properties stored as the 
            individual rows.
        
        features : list
            List of features to be included in the RF model
        
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
            filt_seeing = np.hypot(raw_ps1_df[filt + 'psfMajorFWHM'], 
                                   raw_ps1_df[filt + 'psfMinorFWHM'])
            feat_norm_list = [filt_seeing**2, filt_seeing**2, filt_seeing**2,
                              filt_seeing, filt_seeing, filt_seeing**0.5]
            
            for feat, feat_norm in zip(normed_feat_list, feat_norm_list):
                ff = filt + feat
                self.X_df_[ff + "norm"] = raw_ps1_df[ff]/feat_norm
        
            psf_det = np.array(raw_ps1_df[filt + 'PSFFlux'] > 0).astype(int)
            kron_det = np.array(raw_ps1_df[filt + 'KronFlux'] > 0).astype(int)
            
            psf_det_flux = psf_det*raw_ps1_df[filt + 'PSFFlux'].values
            kron_det_flux = kron_det*raw_ps1_df[filt + 'KronFlux'].values
            
            psf_flux_w += psf_det_flux
            kron_flux_w += kron_det_flux
            
            self.X_df_[filt + 'psfKronRatio'] = np.nan_to_num(psf_det_flux / 
                                                              kron_det_flux)
            
        self.X_df_['wpsfKronRatio'] = np.nan_to_num(psf_flux_w/kron_flux_w)
            
            