# The PanSTARRS1 Point Source Catalog (PS1 PSC)

This repository includes several jupyter notebooks and [MAST Query/CasJobs](http://mastweb.stsci.edu/ps1casjobs/) SQL queries, which were used to generate the [PanSTARRS1 Point Source Catalog (PS1 PSC)](https://archive.stsci.edu/prepds/ps1-psc/).

The results of this work have been written up in two different papers: 

[A Morphological Classification Model to Identify Unresolved PanSTARRS1 Sources: Application in the ZTF Real-time Pipeline](https://doi.org/10.1088/1538-3873/aae3d9). Tachibana & Miller, 2018, PASP, 130, 994

and 

[A Morphological Classification Model to Identify Unresolved PanSTARRS1 Sources II: Update to the PS1 Point Source Catalog](https://arxiv.org/abs/2012.01544). Miller & Hall, 2021, PASP, accepted

The repo is generally organized as follows:
  - notebooks and queries for the original analysis of the PS1 data are in [PS1casjobs/](PS1casjobs/)
  - notebooks showing how well the model performs on gaia data are in [gaia/](gaia/)
  - the source material for the first paper is in [paper/](paper/)
  - notebooks and queries for the PS1 PSC "update" are in [catalog_update/](catalog_update/)
  - the source material for the second paper is in [paperII/](paperII/)