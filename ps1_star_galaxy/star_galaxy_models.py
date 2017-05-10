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
                            training_set = False, target_name = 'class', 
                            random_state = None):
        """
        Generate training set attribute from fits file
        
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
            feat_norm_terms[feat_norm_terms == 0] = -0.0    
                        
            ff_list = [filt + feat for feat in normed_feat_list]
            ff_norm_list = [filt + feat + "norm" for feat in normed_feat_list]

            ff_kwargs = {x : raw_ps1_df[y]/feat_norm_terms[:,z] 
                for x, y, z, in zip(ff_norm_list, ff_list, range(6))}
            
            self.X_df_.assign(**ff_kwargs)
        
            psf_det = np.array(raw_ps1_df[filt + 'PSFFlux'] > 0).astype(int)
            kron_det = np.array(raw_ps1_df[filt + 'KronFlux'] > 0).astype(int)
            
            psf_det_flux = psf_det*raw_ps1_df[filt + 'PSFFlux'].values
            kron_det_flux = kron_det*raw_ps1_df[filt + 'KronFlux'].values
            
            psf_flux_w += psf_det_flux
            kron_flux_w += kron_det_flux
            
            f_kwargs = {filt + 'psfKronRatio' : np.nan_to_num(psf_det_flux / 
                                                              kron_det_flux)}
            self.X_df_.assign(**f_kwargs)
            
        w_kwargs = {'wpsfKronRatio' : np.nan_to_num(psf_flux_w/kron_flux_w)}
        self.X_df_.assign(**w_kwargs)
            
            