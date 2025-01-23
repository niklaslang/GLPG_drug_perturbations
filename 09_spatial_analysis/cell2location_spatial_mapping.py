#!/usr/bin/env python
# coding: utf-8

# ### Running cell2location to map our 10X Genomics Bleo day21 data to publicly available 10X Visium Bleo day21 data

# ST data from [Franzen, Lindvall, et al. Nature Genetics (2024)](https://www.nature.com/articles/s41588-024-01819-2)

# In[1]:


import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb


# In[2]:


import cell2location as c2l


# In[3]:


# making sure plots & clusters are reproducible
np.random.seed(42)


# In[4]:


## plotting variables
sc.settings.figdir = '/home/niklas/projects/GLPG_drug_perturbations/02_figures/fig_spatial_overview/'
sc.set_figure_params(vector_friendly = True)
plt.rcParams['figure.figsize'] = (6, 5)
plt.rcParams['pdf.fonttype'] = 42


# In[5]:


## path variables
sc_dir = '/mnt/smb/Niklas/20240911_10XVisum_Stockholm_mouse/reference_signatures/240923_GLPG_PBS_Bleo_d21_reference_signatures.h5ad'
spatial_dir = '/mnt/smb/Niklas/20240911_10XVisum_Stockholm_mouse/240919_10XVisium_d21.h5ad'
data_dir = '/mnt/smb/Niklas/20240911_10XVisum_Stockholm_mouse/'


# In[6]:


## paths reference regression and cell2location models
ref_run_name = f'{data_dir}/reference_signatures'
run_name = f'{data_dir}/cell2location_map'


# ### Load Visium data

# In[7]:


adata_st = sc.read(spatial_dir)


# ### Load scRNA-seq reference data

# In[8]:


adata_sc = sc.read(sc_dir)


# In[9]:


adata_sc


# ### Load model

# In[10]:


## default, try on GPU:
#use_gpu = False
#model = c2l.models.RegressionModel.load(f'{ref_run_name}', adata_sc)


# ### Extract estimated gene expression per cell type

# In[11]:


# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_sc.varm.keys():
    inf_aver = adata_sc.varm['means_per_cluster_mu_fg'][
        [f'means_per_cluster_mu_fg_{i}' for i in adata_sc.uns['mod']['factor_names']]
    ].copy()
else:
    inf_aver = adata_sc.var[
        [f'means_per_cluster_mu_fg_{i}' for i in adata_sc.uns['mod']['factor_names']]
    ].copy()

inf_aver.columns = adata_sc.uns['mod']['factor_names']
inf_aver.head()


# In[12]:


## 
inf_aver.to_csv(ref_run_name + 'inf_aver.csv')


# ### Cell type mapping

# In[13]:


## find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(adata_st.var_names, inf_aver.index)
adata_st = adata_st[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()


# In[14]:


## prepare anndata
c2l.models.Cell2location.setup_anndata(
    adata=adata_st,
    batch_key='sample',
)


# In[15]:


## create model
model = c2l.models.Cell2location(
    adata_st,
    cell_state_df = inf_aver,
    N_cells_per_location = 7,
    detection_alpha = 20
)
model.view_anndata_setup()


# In[16]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[ ]:


## train model
model.train(max_epochs=30000, batch_size=None, train_size=1, accelerator = 'cpu')
# plot training history
model.plot_history()


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[ ]:


adata_st = model.export_posterior(
    adata_st,
    sample_kwargs={
        'num_samples': 1000,
        'batch_size': model.adata.n_obs
        #'use_gpu': False,
    },
)


# In[ ]:


model.plot_QC()


# In[ ]:


adata_st.obs[adata_st.uns['mod']['factor_names']] = adata_st.obsm[
    'q05_cell_abundance_w_sf'
]


# In[ ]:


## save anndata object with results
adata_file = f"{run_name}/250111_10XVisium_d21_deconvolution.h5ad"
adata_st.write(adata_file)
adata_file


# In[ ]:


#def select_slide(adata, s, s_col='sample'):
#    r""" This function selects the data for one slide from the spatial anndata object.
#
#    :param adata: Anndata object with multiple spatial experiments
#    :param s: name of selected experiment
#    :param s_col: column in adata.obs listing experiment name for each location
#    """
#
#    slide = adata[adata.obs[s_col].isin([s]), :]
#    s_keys = list(slide.uns['spatial'].keys())
#    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]
#
#    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}
#
#    return slide


# In[ ]:


## select one slide for visualization
#for sample_name in adata_st.obs['sample'].unique():
#    slide = select_slide(adata_st, sample_name)
#
#    with mpl.rc_context({"figure.figsize": [4.5, 5]}):
#        sc.pl.spatial(
#            slide,
#            cmap="magma",
#            color=adata_st.uns["mod"]["factor_names"],
#            ncols=4,
#            size=1.3,
#            img_key="hires",
#            # limit color scale at 99.2% quantile of cell abundance
#            vmin=0,
#            vmax="p99.2",
#        )


# In[ ]:


#clust_col = ['Eosinophils','Cthrc1= Myofibroblasts','Krt8+ ADI','Vwa1+/Col15a1+ ectopic EC']
#clust_labels = clust_col
#
## select one slide for visualization
#for sample_name in adata_st.obs['sample'].unique():
#    slide = select_slide(adata_st, sample_name)
#    with matplotlib.rc_context({"figure.figsize": (15, 15)}):
#    fig = c2l.plt.plot_spatial(
#        adata=slide,
#        color=clust_col,
#        labels=clust_labels,
#        max_color_quantile=0.992,
#        circle_diameter=6,
#        show_img=True,
#        colorbar_position="right",
#        colorbar_shape={"horizontal_gaps": 0.2},
#    )

