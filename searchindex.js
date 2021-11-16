Search.setIndex({docnames:["cli","contributing","dataframe_transforms","dynamic_imports","example_workflows","feature_extraction","image_transforms","index","installation","modules","quickstart","serotiny","serotiny.data","serotiny.data.dataframe","serotiny.data.image","serotiny.datamodules","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.loaders","serotiny.losses","serotiny.metrics","serotiny.models","serotiny.models.callbacks","serotiny.models.vae","serotiny.networks","serotiny.networks.classification","serotiny.networks.layers","serotiny.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","contributing.rst","dataframe_transforms.rst","dynamic_imports.rst","example_workflows.rst","feature_extraction.rst","image_transforms.rst","index.rst","installation.rst","modules.rst","quickstart.rst","serotiny.rst","serotiny.data.rst","serotiny.data.dataframe.rst","serotiny.data.image.rst","serotiny.datamodules.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.loaders.rst","serotiny.losses.rst","serotiny.metrics.rst","serotiny.models.rst","serotiny.models.callbacks.rst","serotiny.models.vae.rst","serotiny.networks.rst","serotiny.networks.classification.rst","serotiny.networks.layers.rst","serotiny.utils.rst"],objects:{"":[[11,0,0,"-","serotiny"]],"serotiny.data":[[13,0,0,"-","dataframe"],[12,0,0,"-","predict"]],"serotiny.data.dataframe":[[13,0,0,"-","transforms"]],"serotiny.data.dataframe.transforms":[[13,1,1,"","append_class_weights"],[13,1,1,"","append_labels_to_integers"],[13,1,1,"","append_one_hot"],[13,1,1,"","filter_columns"],[13,1,1,"","filter_rows"],[13,1,1,"","make_random_df"],[13,1,1,"","sample_n_each"],[13,1,1,"","split_dataframe"]],"serotiny.data.image":[[14,0,0,"-","pad"],[14,0,0,"-","project"],[14,0,0,"-","resize"],[14,0,0,"-","swap"]],"serotiny.data.image.pad":[[14,2,1,"","ExpandColumns"],[14,2,1,"","ExpandTo"],[14,2,1,"","PadTo"],[14,1,1,"","expand_columns"],[14,1,1,"","expand_to"],[14,1,1,"","pull_to"],[14,1,1,"","split_number"],[14,1,1,"","to_tensor"]],"serotiny.data.image.project":[[14,2,1,"","Project"]],"serotiny.data.image.resize":[[14,2,1,"","CropCenter"],[14,2,1,"","ResizeBy"],[14,2,1,"","ResizeTo"]],"serotiny.data.image.swap":[[14,2,1,"","Permute"],[14,2,1,"","SwapAxes"]],"serotiny.data.predict":[[12,1,1,"","tile_prediction"]],"serotiny.datamodules":[[15,2,1,"","DummyDatamodule"],[15,2,1,"","ManifestDatamodule"],[15,2,1,"","PatchDatamodule"],[15,0,0,"-","dummy"],[15,0,0,"-","gaussian"],[15,0,0,"-","manifest_datamodule"],[15,0,0,"-","patch"]],"serotiny.datamodules.DummyDatamodule":[[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.ManifestDatamodule":[[15,3,1,"","make_dataloader"],[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.PatchDatamodule":[[15,3,1,"","load_patch_manifest"],[15,3,1,"","make_dataloader"],[15,3,1,"","make_patch_dataset"],[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.dummy":[[15,2,1,"","DummyDatamodule"],[15,2,1,"","DummyDataset"],[15,1,1,"","make_dataloader"]],"serotiny.datamodules.dummy.DummyDatamodule":[[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.gaussian":[[15,2,1,"","GaussianDataModule"],[15,2,1,"","GaussianDataset"],[15,1,1,"","make_dataloader"]],"serotiny.datamodules.gaussian.GaussianDataModule":[[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.gaussian.GaussianDataset":[[15,3,1,"","random_corr_mat"]],"serotiny.datamodules.manifest_datamodule":[[15,2,1,"","ManifestDatamodule"]],"serotiny.datamodules.manifest_datamodule.ManifestDatamodule":[[15,3,1,"","make_dataloader"],[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.datamodules.patch":[[15,2,1,"","PatchDatamodule"],[15,1,1,"","make_manifest_dataset"]],"serotiny.datamodules.patch.PatchDatamodule":[[15,3,1,"","load_patch_manifest"],[15,3,1,"","make_dataloader"],[15,3,1,"","make_patch_dataset"],[15,3,1,"","test_dataloader"],[15,3,1,"","train_dataloader"],[15,3,1,"","val_dataloader"]],"serotiny.io":[[16,0,0,"-","buffered_patch_dataset"],[17,0,0,"-","dataframe"],[16,0,0,"-","image"],[16,0,0,"-","quilt"]],"serotiny.io.buffered_patch_dataset":[[16,2,1,"","BufferedPatchDataset"]],"serotiny.io.buffered_patch_dataset.BufferedPatchDataset":[[16,3,1,"","get_buffer_history"],[16,3,1,"","get_patch"],[16,3,1,"","get_random_patch"],[16,3,1,"","insert_new_element_into_buffer"]],"serotiny.io.dataframe":[[17,0,0,"-","dataframe_dataset"],[18,0,0,"-","loaders"],[17,0,0,"-","readers"]],"serotiny.io.dataframe.dataframe_dataset":[[17,2,1,"","DataframeDataset"]],"serotiny.io.dataframe.loaders":[[18,2,1,"","Load2DImage"],[18,2,1,"","Load3DImage"],[18,2,1,"","LoadClass"],[18,2,1,"","LoadColumn"],[18,2,1,"","LoadColumns"],[18,0,0,"-","abstract_loader"],[18,0,0,"-","classes"],[18,0,0,"-","columns"],[18,0,0,"-","image2d"],[18,0,0,"-","image3d"]],"serotiny.io.dataframe.loaders.abstract_loader":[[18,2,1,"","Loader"]],"serotiny.io.dataframe.loaders.classes":[[18,2,1,"","LoadClass"]],"serotiny.io.dataframe.loaders.columns":[[18,2,1,"","LoadColumn"],[18,2,1,"","LoadColumns"]],"serotiny.io.dataframe.loaders.image2d":[[18,2,1,"","Load2DImage"]],"serotiny.io.dataframe.loaders.image3d":[[18,2,1,"","Load3DImage"]],"serotiny.io.dataframe.readers":[[17,1,1,"","filter_columns"],[17,1,1,"","read_csv"],[17,1,1,"","read_dataframe"],[17,1,1,"","read_parquet"]],"serotiny.io.image":[[16,1,1,"","image_loader"],[16,1,1,"","infer_dims"],[16,1,1,"","tiff_writer"]],"serotiny.io.quilt":[[16,1,1,"","download_quilt_data"]],"serotiny.losses":[[19,0,0,"-","continuous_bernoulli"],[19,0,0,"-","kl_divergence"]],"serotiny.losses.continuous_bernoulli":[[19,2,1,"","CBLogLoss"]],"serotiny.losses.continuous_bernoulli.CBLogLoss":[[19,3,1,"","forward"],[19,4,1,"","reduction"]],"serotiny.losses.kl_divergence":[[19,1,1,"","diagonal_gaussian_kl"],[19,1,1,"","isotropic_gaussian_kl"]],"serotiny.metrics":[[20,2,1,"","InceptionV3"],[20,1,1,"","calculate_blur"],[20,0,0,"-","calculate_blur"],[20,1,1,"","calculate_fid"],[20,0,0,"-","calculate_fid"],[20,0,0,"-","inception"],[20,0,0,"-","pearson"]],"serotiny.metrics.InceptionV3":[[20,4,1,"","BLOCK_INDEX_BY_DIM"],[20,4,1,"","DEFAULT_BLOCK_INDEX"],[20,3,1,"","forward"],[20,4,1,"","training"]],"serotiny.metrics.calculate_blur":[[20,1,1,"","calculate_blur"]],"serotiny.metrics.calculate_fid":[[20,1,1,"","calculate_fid"],[20,1,1,"","get_activations"]],"serotiny.metrics.inception":[[20,2,1,"","InceptionV3"]],"serotiny.metrics.inception.InceptionV3":[[20,4,1,"","BLOCK_INDEX_BY_DIM"],[20,4,1,"","DEFAULT_BLOCK_INDEX"],[20,3,1,"","forward"],[20,4,1,"","training"]],"serotiny.metrics.pearson":[[20,1,1,"","pearson_correlation"]],"serotiny.models":[[21,2,1,"","ClassificationModel"],[21,2,1,"","ImageVAE"],[21,2,1,"","RegressionModel"],[21,2,1,"","TabularConditionalPriorVAE"],[21,2,1,"","TabularConditionalVAE"],[21,2,1,"","TabularVAE"],[21,2,1,"","UnetModel"],[22,0,0,"-","callbacks"],[21,0,0,"-","classification"],[21,0,0,"-","regression"],[21,0,0,"-","unet"],[21,0,0,"-","utils"],[23,0,0,"-","vae"],[21,0,0,"-","zoo"]],"serotiny.models.ClassificationModel":[[21,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","on_after_backward"],[21,3,1,"","parse_batch"],[21,4,1,"","precision"],[21,4,1,"","prepare_data_per_node"],[21,3,1,"","test_epoch_end"],[21,3,1,"","test_step"],[21,3,1,"","test_step_end"],[21,4,1,"","training"],[21,3,1,"","training_step"],[21,3,1,"","training_step_end"],[21,4,1,"","use_amp"],[21,3,1,"","validation_epoch_end"],[21,3,1,"","validation_step"],[21,3,1,"","validation_step_end"]],"serotiny.models.ImageVAE":[[21,4,1,"","training"]],"serotiny.models.RegressionModel":[[21,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","parse_batch"],[21,4,1,"","precision"],[21,4,1,"","prepare_data_per_node"],[21,3,1,"","test_epoch_end"],[21,3,1,"","test_step"],[21,4,1,"","training"],[21,3,1,"","training_epoch_end"],[21,3,1,"","training_step"],[21,4,1,"","use_amp"],[21,3,1,"","validation_epoch_end"],[21,3,1,"","validation_step"]],"serotiny.models.TabularConditionalPriorVAE":[[21,4,1,"","training"]],"serotiny.models.TabularConditionalVAE":[[21,3,1,"","parse_batch"],[21,4,1,"","training"]],"serotiny.models.TabularVAE":[[21,4,1,"","training"]],"serotiny.models.UnetModel":[[21,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","get_unet_padding"],[21,3,1,"","on_after_backward"],[21,3,1,"","parse_batch"],[21,4,1,"","precision"],[21,4,1,"","prepare_data_per_node"],[21,3,1,"","test_step"],[21,3,1,"","test_step_end"],[21,4,1,"","training"],[21,3,1,"","training_step"],[21,4,1,"","use_amp"],[21,3,1,"","validation_step"]],"serotiny.models.callbacks":[[22,0,0,"-","plot_xhat"]],"serotiny.models.callbacks.plot_xhat":[[22,2,1,"","PlotXHat"]],"serotiny.models.callbacks.plot_xhat.PlotXHat":[[22,3,1,"","on_validation_batch_end"]],"serotiny.models.classification":[[21,2,1,"","ClassificationModel"],[21,1,1,"","acc_prec_recall"]],"serotiny.models.classification.ClassificationModel":[[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","on_after_backward"],[21,3,1,"","parse_batch"],[21,3,1,"","test_epoch_end"],[21,3,1,"","test_step"],[21,3,1,"","test_step_end"],[21,4,1,"","training"],[21,3,1,"","training_step"],[21,3,1,"","training_step_end"],[21,3,1,"","validation_epoch_end"],[21,3,1,"","validation_step"],[21,3,1,"","validation_step_end"]],"serotiny.models.regression":[[21,2,1,"","RegressionModel"]],"serotiny.models.regression.RegressionModel":[[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","parse_batch"],[21,3,1,"","test_epoch_end"],[21,3,1,"","test_step"],[21,4,1,"","training"],[21,3,1,"","training_epoch_end"],[21,3,1,"","training_step"],[21,3,1,"","validation_epoch_end"],[21,3,1,"","validation_step"]],"serotiny.models.unet":[[21,2,1,"","UnetModel"]],"serotiny.models.unet.UnetModel":[[21,3,1,"","configure_optimizers"],[21,3,1,"","forward"],[21,3,1,"","get_unet_padding"],[21,3,1,"","on_after_backward"],[21,3,1,"","parse_batch"],[21,3,1,"","test_step"],[21,3,1,"","test_step_end"],[21,4,1,"","training"],[21,3,1,"","training_step"],[21,3,1,"","validation_step"]],"serotiny.models.utils":[[21,1,1,"","add_pr_curve_tensorboard"],[21,1,1,"","find_lr_scheduler"],[21,1,1,"","find_optimizer"],[21,1,1,"","index_to_onehot"]],"serotiny.models.vae":[[23,0,0,"-","base_vae"],[23,0,0,"-","conditional_vae"],[23,0,0,"-","condprior_vae"],[23,0,0,"-","image_vae"],[23,0,0,"-","tabular_conditional_vae"],[23,0,0,"-","tabular_condprior_vae"],[23,0,0,"-","tabular_vae"]],"serotiny.models.vae.base_vae":[[23,2,1,"","BaseVAE"]],"serotiny.models.vae.base_vae.BaseVAE":[[23,3,1,"","calculate_elbo"],[23,3,1,"","configure_optimizers"],[23,3,1,"","forward"],[23,3,1,"","parse_batch"],[23,3,1,"","sample_z"],[23,3,1,"","test_step"],[23,4,1,"","training"],[23,3,1,"","training_step"],[23,3,1,"","validation_step"]],"serotiny.models.vae.conditional_vae":[[23,2,1,"","ConditionalVAE"]],"serotiny.models.vae.conditional_vae.ConditionalVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,3,1,"","parse_batch"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.vae.condprior_vae":[[23,2,1,"","ConditionalPriorVAE"]],"serotiny.models.vae.condprior_vae.ConditionalPriorVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,3,1,"","forward"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.vae.image_vae":[[23,2,1,"","ImageVAE"]],"serotiny.models.vae.image_vae.ImageVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.vae.tabular_conditional_vae":[[23,2,1,"","TabularConditionalVAE"]],"serotiny.models.vae.tabular_conditional_vae.TabularConditionalVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,3,1,"","parse_batch"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.vae.tabular_condprior_vae":[[23,2,1,"","TabularConditionalPriorVAE"]],"serotiny.models.vae.tabular_condprior_vae.TabularConditionalPriorVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.vae.tabular_vae":[[23,2,1,"","TabularVAE"]],"serotiny.models.vae.tabular_vae.TabularVAE":[[23,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[23,4,1,"","precision"],[23,4,1,"","prepare_data_per_node"],[23,4,1,"","training"],[23,4,1,"","use_amp"]],"serotiny.models.zoo":[[21,1,1,"","get_checkpoint_callback"],[21,1,1,"","get_model"],[21,1,1,"","get_root"],[21,1,1,"","get_trainer_at_checkpoint"],[21,1,1,"","store_metadata"],[21,1,1,"","store_model"]],"serotiny.networks":[[25,0,0,"-","classification"],[26,0,0,"-","layers"],[24,0,0,"-","mlp"],[24,0,0,"-","sequential"],[24,0,0,"-","weight_init"]],"serotiny.networks.layers":[[26,0,0,"-","activation"],[26,0,0,"-","pad"],[26,0,0,"-","spatial_pyramid_pool"]],"serotiny.networks.layers.activation":[[26,1,1,"","activation_map"]],"serotiny.networks.layers.pad":[[26,2,1,"","PadLayer"]],"serotiny.networks.layers.pad.PadLayer":[[26,3,1,"","forward"],[26,4,1,"","training"]],"serotiny.networks.layers.spatial_pyramid_pool":[[26,2,1,"","SpatialPyramidPool"],[26,1,1,"","spatial_pyramid_pool"]],"serotiny.networks.layers.spatial_pyramid_pool.SpatialPyramidPool":[[26,3,1,"","forward"],[26,4,1,"","training"]],"serotiny.networks.mlp":[[24,2,1,"","MLP"]],"serotiny.networks.mlp.MLP":[[24,3,1,"","forward"],[24,4,1,"","training"]],"serotiny.networks.sequential":[[24,2,1,"","Sequential"]],"serotiny.networks.sequential.Sequential":[[24,3,1,"","forward"],[24,4,1,"","training"]],"serotiny.networks.weight_init":[[24,1,1,"","weight_init"]],"serotiny.utils":[[27,0,0,"-","dynamic_imports"]],"serotiny.utils.dynamic_imports":[[27,1,1,"","bind"],[27,1,1,"","get_load_method_and_args"],[27,1,1,"","get_name_and_arguments"],[27,1,1,"","get_name_from_path"],[27,1,1,"","init"],[27,1,1,"","invoke"],[27,1,1,"","load_config"],[27,1,1,"","load_multiple"],[27,1,1,"","module_get"],[27,1,1,"","module_or_path"],[27,1,1,"","module_path"],[27,1,1,"","search_modules"]],serotiny:[[12,0,0,"-","data"],[15,0,0,"-","datamodules"],[11,1,1,"","get_module_version"],[16,0,0,"-","io"],[19,0,0,"-","losses"],[20,0,0,"-","metrics"],[21,0,0,"-","models"],[24,0,0,"-","networks"],[27,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"0":[2,3,12,14,15,19,20,21,23],"001":[21,23],"00638175":3,"0151588":3,"01525316":3,"05956044":3,"06":20,"06845":19,"1":[2,3,5,13,15,16,19,20,21,23],"10":[0,3,6,15],"100":[5,6,13],"14726909":3,"19":3,"1907":19,"192":20,"1e":[20,21],"2":[2,3,20],"20":3,"2048":20,"20821983":3,"256":24,"29300648":3,"299":20,"2d":[18,21,23,26],"3":[2,3,20,21],"32":16,"32604151":3,"32854592":3,"3287971":3,"344":20,"37383497":3,"3d":[12,18,21,22,23,26],"4":[0,2,3],"41411613":3,"42":13,"48772363":3,"49404687":3,"4d":12,"5":3,"50":2,"54736776":3,"56047808":3,"59323792":3,"59621":3,"60767339":3,"64":[3,12,16,20],"66560783":3,"68928192":3,"7":[2,3],"768":20,"78241693":3,"78599943":3,"8":3,"82680974":3,"8956135":3,"89774285":3,"98165106":3,"98437869":3,"98945791":3,"98970493":3,"abstract":18,"boolean":[21,23],"break":[0,5,6],"case":[3,6,13,15,21,23],"class":[3,13,14,15,16,17,19,20,21,22,23,24,26,27],"default":13,"do":0,"final":[0,6,20],"float":[13,18,20,21,23],"function":[0,3,5,6,13,15,17,20,21,23,26,27],"import":[5,6,10,18,21],"int":[12,13,14,15,16,18,20,21,23,26],"new":[1,16],"public":8,"return":[3,5,6,11,12,13,15,16,17,18,20],"short":1,"true":[5,6,13,14,15,16,20,21,27],"try":[5,6],A:[1,6,13,15,17,18,20,21],AND:17,As:[5,6,20],But:18,By:13,For:[0,2,3,5,6,17,18],If:[0,3,8,13,15,16,17,18,20,21,23],In:[6,13,17,21,23],It:[1,2,6,13,15,18,19],Or:8,That:2,The:[0,3,5,6,8,13,16,17,18,20,21],Then:1,There:3,To:[2,3,6,8],_3d:3,_get_checkpoint:10,_loss:[19,21,23],_val:21,ab:19,abl:20,abov:0,abstract_load:[16,17],acc_prec_recal:21,accept:0,accord:26,activ:[11,20,24],activation_map:26,ad:0,adam:[21,23],adapt:3,add:[0,1,13],add_pr_curve_tensorboard:21,addit:21,addition:[3,6],after:0,agglomer:11,ahead:[3,5,6],aic:[11,16,17],aics_imag:16,aicsimag:16,aicsimageio:[5,6,16],align:[11,12],all:[0,1,2,3,6,10,13,15,17,18,20,21],allencel:16,allencellmodel:8,allow:[2,3,6,24],allow_zero_length_dataloader_with_multiple_devic:[21,23],almost:3,along:12,also:[0,1,2,3,6,17,19],altern:15,alwai:[1,8,18],an:[0,3,5,6,11,12,16,18,19,20,21,23],anaconda:1,ani:[0,6,15],anisotrop:[21,23],api:[5,21],append:[0,13],append_class_weight:13,append_labels_to_integ:13,append_one_hot:13,appli:[0,16,17,18,21,24,26],appreci:1,appropri:17,ar:[1,3,6,13,17,18,20],arbitrari:20,aren:17,arg:15,argument:[0,2,3,6,13,17,21],arrai:[3,14,16,18,20,21,23],arraylik:16,arxiv:19,ascend:20,assert:17,assign:16,associ:21,assum:10,attrib:16,attribut:[3,16],auto_pad:21,autoencod:[21,23],autograd:20,aux:20,auxiliari:24,avail:[18,21],averag:20,avoid:[5,6],ax:14,axi:14,b:[1,2,13,20],backend:[5,6],balanc:13,base:[3,13,14,15,16,17,18,19,20,21,22,23,24,26],base_va:[11,21],baseva:[21,23],basic:23,batch:[15,19,20,21,22,23],batch_idx:[21,22,23],batch_siz:[15,20],befor:[16,20],behaviour:6,belong:13,bernoulli:19,beta:[15,21,23],better:15,between:[13,16,19,20],bilinearli:20,bimod:15,binari:18,bind:27,binomi:15,bit:1,blob:20,block:20,block_index_by_dim:20,blurri:20,bool:[13,15,16,18,20,21,23,24,26],both:6,bound:5,bounding_box:5,box:5,br:1,branch:1,bucket:16,buffer:16,buffer_index:16,buffer_s:16,buffer_switch_interv:16,buffered_patch_dataset:[9,11],bufferedpatchdataset:16,bugfix:1,build:[1,20],bump2vers:1,bx3xhxw:20,c:[2,13],c_1:20,c_2:20,c_dim:[21,23],c_label:[15,21,23],c_label_ind:15,cach:21,calcul:20,calculate_blur:[9,11],calculate_elbo:23,calculate_fid:[9,11],call:[3,5,6,13],callabl:[5,16,18,21,23],callback:[11,21],can:[0,1,2,3,5,6,8,13,15,23],cblogloss:19,cd:1,cell:[11,17],cell_id:[6,22],cell_id_label:22,cellid:[2,5],center_of_mass:14,centric:[3,6],certain:5,chang:1,channel:[3,6,12,15,16,18,21,23],channel_fan:[3,21],channel_fan_top:3,channel_nam:16,check:[1,21],checkout:1,checkpoint_callback_kwarg:21,chunk:0,class_index:21,classif:[9,11,24],classifi:[20,21],classificationmodel:21,classnam:27,cli:[2,5,6],clone:[1,8],coeffici:[5,20],col:6,collat:15,colon:0,column:[2,5,6,13,14,15,16,17,21],columns_to_filt:17,com:[1,8,20],combin:[12,17],come:0,command:[0,6,8,21],commit:1,common:[2,6],complet:11,complex:[0,6],composit:24,comput:[5,19],concaten:[6,18],condit:[15,17,21,23],condition_mod:[21,23],conditional_va:[11,21],conditionalpriorva:[21,23],conditionalva:[21,23],condprior_va:[11,21],config:[3,5,6,10,24,27],configur:[0,3,18,21],configure_optim:[21,23],conform:5,connect:20,consist:17,constant:14,contain:[0,3,5,6,10,13,15,16,17,18,20,21],content:9,continu:[5,6,19],continuous_bernoulli:[9,11],conv:21,conveni:6,convert:13,convolut:20,copi:8,core:[13,15,17,21,23],corr:15,correct:16,correl:[15,20],correspond:[5,6,13,16,17,20,21],covari:[20,21,23],creat:[1,2,13,15],credit:1,crop_and_norm:6,crop_raw:2,crop_seg:2,cropcent:14,cropi:14,cropx:14,cropz:14,csv:[2,5,6,17],cuda:[12,20],cumbersom:0,curl:8,current:[21,23],curv:21,d:[13,15,20],dash:0,data:[6,9,11,15,16,17,18,20,26],data_save_loc:16,datafram:[0,6,11,12,16],dataframe_dataset:[11,16],dataframedataset:[17,18],dataload:15,dataloader_idx:22,datamodul:[9,10,11],dataset:[5,6,13,15,16,17],debug:[5,6],decod:[21,23],decoder_non_linear:[21,23],default_block_index:20,defin:[0,2],depend:[6,15,20],depth:[3,21],describ:[3,6,21,23],descript:1,desir:[0,17],detail:[1,13],determin:[16,18,21,23],dev:1,develop:[1,3],devic:12,diagon:[15,19,21,23],diagonal_gaussian_kl:19,dict:[5,6,15,17,18,21,23,24],dictionari:[0,3,5,6,15],differ:[0,6,13,17],dillat:6,dim:[15,20,24],dim_ord:16,dimens:[12,14,15,16,20,21,26],dimension:[20,26],dims_max:12,direct:5,directli:[6,18],disabl:16,disk:[5,6,16,21],distanc:20,dive:6,diverg:19,doc:21,doesn:[16,21],don:[0,8],done:[1,2],dot:0,doubl:21,dougal:20,down:[3,21],download:[8,16],download_quilt_data:16,downstream:5,drop:15,drop_last:15,dtype:[3,16,18],dummi:[9,11],dummydatamodul:15,dummydataset:15,dynam:[5,6,18],dynamic_import:[9,11],e:[1,5,6,17,21,23,27],each:[3,5,6,12,13,15,16,17,20,21,26],easi:3,edit:1,either:[5,6,8,13,15,17,21,23],element:16,en:21,enabl:[5,6,13,18],encod:[13,21,23],end:[0,2,13,17,18],endswith:[13,17,18],enough:[0,3],ensur:21,enter:0,environ:[1,21],ep:20,equal:13,equival:6,error:[5,6],etc:5,everi:[1,6],everyth:[4,15],ex:1,exampl:[3,16,21],exchang:16,exclud:[2,13,17,18],exist:[5,13,21],expand_column:14,expand_to:14,expandcolumn:14,expanded_column:14,expandto:14,expect:[3,6,17,20,21,23],experi:2,explain:5,explicit:[13,18],expos:2,express:[13,17,18],extend:13,extract:[16,18],extract_featur:5,factor:14,fals:[2,5,6,13,15,16,18,20,21,22,23],featur:[1,20],feature_extract:[0,11,12],fed:20,feed:20,field:[0,6,18],file:[0,1,2,3,5,6,10,15,16,17,18],file_typ:18,filenam:[0,6],filter:[2,6,13,17,18],filter_column:[2,13,17,18],filter_row:[2,13],find:5,find_lr_schedul:21,find_optim:21,finetun:20,first:[0,2,12,14,19,20],fix:[21,23],flag:[16,18,21,23],flexibl:6,float32:16,fold:15,folder:21,forc:13,force_s:14,fork:1,format:18,forward:[19,20,21,23,24,26],found:[1,17],fraction:13,frame:[13,17],framework:3,frechet:20,from:[3,6,10,15,16,17,18,21,23,24,26],full_load:3,fulli:20,further:3,futur:18,g:[5,6,21,27],gaussian:[9,11,19,20],gaussiandatamodul:15,gaussiandataset:15,gener:[2,5,6,13,15,20,21],get:[20,21],get_activ:20,get_buffer_histori:16,get_checkpoint_callback:21,get_load_method_and_arg:27,get_model:[10,21],get_module_vers:11,get_name_and_argu:27,get_name_from_path:27,get_patch:16,get_predict:20,get_random_patch:16,get_root:21,get_trainer_at_checkpoint:[10,21],get_unet_pad:21,gh:1,git:[1,8],github:[1,8,20],give:15,given:[1,6,13,15,16,17,18,20,21,23,26,27],global_step:21,gpu:[15,20],gradient:20,greatli:1,guid:8,ha:[0,3,13,16,21],half:13,handl:[1,15,20],hardwar:20,harmon:5,have:[0,8,13,16],height:20,help:[0,1],helper:2,here:[1,2,6,19,21],hi:20,hidden_channel:[21,23],hidden_lay:[21,23,24],high:3,hipsc_single_cell_image_dataset:16,hot:[13,18],how:[0,1,3,4],howev:[0,6],html:[1,21],http:[8,19,20,21],i:[0,6,17,23],idea:3,identifi:[13,21],ignor:[13,17,18],ignore_warn:16,imag:[0,9,11,12,15,17,18,20,21],image2d:[16,17],image3d:[16,17],image_load:16,image_read:[5,6],image_va:[11,21],imageva:[21,23],img:16,implement:[2,18,20,21,23],importantli:2,in_channel:[21,23],incept:[9,11],inceptionv3:20,includ:[1,5,6,13,17,18],include_col:[5,6],include_column:17,index:[6,7,16,20,21],index_col:[5,6],index_to_onehot:21,indic:[15,16,20,21],individual_arg:6,infer:[0,16],infer_dim:16,inform:0,init:27,initi:[5,6,21,23],inp:20,input:[0,2,5,6,12,13,15,19,20,21,23],input_channel:[0,21],input_dim:[21,23],input_manifest:[5,6],insert:16,insert_new_element_into_buff:16,insid:21,instal:1,instanc:[18,20,21],instanti:[3,15,21,23,24],instead:[5,6],integ:[13,16,21,23],integr:3,intend:3,interest:[3,6],interfac:23,intermedi:[5,6],interpret:[0,3],invok:27,involv:2,io:[5,6,9,11,21],isotrop:[19,21,23],isotropic_gaussian_kl:19,item:16,iter:[5,6],its:[3,6,13,18],j:20,just:[0,3],keep:[13,20],kei:[5,6,15,17,21,27],keyword:[3,6,21],kl_diverg:[9,11],kld:[21,23],know:[0,2,6],kullback:19,kwarg:[12,15,23],l:3,label:[15,16,21,23],larger:15,last:[6,15,16],latenc:17,latent_dim:[21,23],latent_dimens:0,layer:[11,20,24],learn:[21,23],learn_prior_logvar:[21,23],least:[5,6,21,23],leibler:19,length:15,less:13,let:[2,3,6,16],leverag:[3,18],lie:20,lightn:[15,21,23],lightningdatamodul:15,lightningmodul:[21,23],like:[3,18,20],line:[0,6,20,21],lint:1,list:[3,5,6,12,13,15,16,17,18,20,21,23],littl:1,load2dimag:18,load3dimag:18,load:[3,5,15,16,17,18,21],load_config:[3,27],load_multipl:27,load_patch_manifest:15,loadabl:0,loadclass:18,loadcolumn:18,loader:[15,16,17],local:1,log:[19,21,23],log_var:19,logger:21,logic:[6,15,17,21],logit:22,logvar1:19,logvar2:19,logvar:[19,23],loop:20,loss:[9,11,21,23],loss_mask_label:[21,23],low:3,lr:[21,23],lr_schedul:21,m0:21,m1:21,m2:21,m2r:1,m3:21,m4:21,m5:21,m6:21,m7:21,m:[0,1,24],made:15,main:8,maintain:1,major:1,make:[0,1,3,6],make_dataload:15,make_manifest_dataset:15,make_patch_dataset:15,make_random_df:[2,13],mani:3,manifest:[2,5,6,15,17,18],manifest_datamodul:[9,11],manifestdatamodul:15,map:[16,20,26],mask:[15,23],master:20,match:[13,17,18,21],matrix:[15,20,21,23],matter:6,max:[14,20],max_pool_lay:[21,23],maxim:3,mean:[19,20,21,23],mechan:18,merg:[0,2,5,6],metadata:21,method:[8,12,14,17],metric:[9,11,21],might:[0,18,20],min:20,minor:1,misc:0,mitotic_class:21,ml:11,mlp:[9,11],mode:[1,14,15,19,21,22,23],model:[0,9,11,20,24],model_checkpoint:21,model_class:21,model_id:[10,21],model_path:10,model_zoo:10,model_zoo_path:10,modelcheckpoint:21,modifi:13,modul:[0,2,5,7,9],module_get:27,module_or_path:27,module_path:27,moduledict:21,more:[0,13,16],moreov:3,most:8,mostli:2,mseloss:[21,23],mu1:[19,20],mu2:[19,20],mu:[19,23],mu_1:20,mu_2:20,multipl:[0,6,18],multiprocess:[5,6],multivari:20,must:[3,6,13,16,17,18,20],my_computed_featur:5,my_input:2,my_manifest:[5,6],my_output:2,my_randint:3,my_randn:3,n:[14,20],n_class:21,n_imag:20,n_row:[2,13],n_worker:[5,6],name:[5,6,15,16,17,18,21],nd:16,ndarrai:[16,20],nearest:14,necessarili:19,need:[2,3,5,6,20],nest:0,net:[3,20],net_config:3,network:[3,9,11,20,21,23],network_config:21,neural:21,nf:16,nn:[19,20,21,23,24,26],none:[3,13,14,15,16,17,18,21,22,23,24],normal:[6,11,12,17,20],normalize_input:20,notat:0,note:[0,2,3,6],noth:18,now:[1,3,5,18],np:16,num:20,num_class:18,num_input_channel:3,num_output_channel:3,num_work:15,number:[3,5,6,13,15,16,18,20,21],numpi:[3,6,16,20,21,23],object:[12,14,16,18],obtain:[0,3,6],off:15,often:5,ol:8,ome_tiff_read:[5,6],ometiffread:[5,6],on_after_backward:21,on_validation_batch_end:22,onc:[5,6,8],one:[0,5,6,10,13,15,18],one_hot_encod:13,ones:[3,5],ones_config:3,onli:[3,13,15,16,17,18,21,23],oper:[0,5,6,13],opt:[21,23],optim:[21,23],optimizer_nam:21,option:[5,6,13,15,16,17,18,21,23],order:[6,16],org:19,origin:[1,3],other:[1,6,18],otherwis:[17,21],our:6,out:21,out_pool_s:26,out_step:20,output:[5,16,20,21,22],output_block:20,output_channel:21,output_dtyp:16,output_manifest:6,output_path:[2,5,6,21],over:20,overlap:12,overrid:[0,21],packag:[1,9],pad:[6,11,12,21,24],pad_dim:26,padlay:26,padto:14,page:[6,7],panda:[13,17,18],parallel:[5,6],param:20,paramet:[0,3,6,12,13,15,16,17,18,19,20,21,23,26],parquet:[15,17],parse_batch:[21,23],part:6,partial:3,partit:[0,2],pass:[1,3,12],patch:[1,9,11,16],patch_column:16,patch_shap:16,patchdatamodul:15,path:[0,5,6,15,16,17,18,21,27],path_col:5,pathlib:[16,17,21],pattern:17,pd:[13,17],pearson:[9,11],pearson_correl:20,per:[0,6,13],perform:[12,13,15],permut:[14,15],piecewis:12,pin_memori:15,pip:[1,8],pipelin:[0,13],placehold:2,plot:21,plot_xhat:[11,21],plotxhat:22,point:[2,5,6,17],pool:20,pool_3:20,posit:3,possibl:[0,1,15,18,20],practic:17,precalcul:20,preced:[2,17],precis:[21,23],predict:[0,9,11],predict_kwarg:12,predictor:12,prefer:[0,8],prepare_data_per_nod:[21,23],pretrain:20,previou:[2,6],primit:11,print:21,prior:[21,23],prior_encod:23,prior_encoder_hidden_lay:[21,23],prior_logvar:[21,23],prior_mod:[21,23],prob:19,problem:21,process:[5,6,8,15],produc:[5,6],project:[1,11,12,15],projection_imag:21,propos:19,provid:[2,3,4,5,6,13,16,18],publish:1,pull:1,pull_to:14,push:1,py:[8,20],pypi:1,python:[1,3,8],pytorch:[15,17,21,23],pytorch_lightn:[15,21,22,23],quantil:5,queri:[17,18,20],quilt:[9,11],quit:0,rais:[5,6],randint:3,randint_config:3,randn:3,randn_config:3,random:[2,3,13,15,16],random_corr_mat:15,rang:20,rate:[21,23],raw:1,re:[1,2],read:[1,3,17],read_csv:17,read_datafram:17,read_parquet:17,reader:[5,6,11,16],readi:1,readthedoc:21,reason:[0,20],recal:21,recent:8,recognit:21,recommend:1,reconstruct:[21,23],reconstruction_loss:[21,23],reconstruction_reduc:[21,23],recurr:27,reduct:19,regex:[13,17,18],regist:6,regress:[9,11],regressionmodel:21,regular:[13,17,18],releas:1,reli:6,reload:21,reload_callback:21,reload_logg:21,remain:[13,17,18],remind:1,replac:13,repo:[1,8],report:20,repositori:8,represent:20,request:1,requir:[6,17,20],required_column:17,requires_grad:20,resiz:[6,11,12,20],resize_input:20,resizebi:14,resizeto:14,resolv:1,respect:[17,23],result:[2,3,5,6,12,13,17,18],retriev:[5,6,15,16,18,21,23,27],retriv:15,return_channel:16,return_merg:[5,6],return_split:[2,13],root:21,row:[2,13,14,18],run:[1,5,6,8,21],s3:16,s:[0,1,3,5,6,13,15,18,21],same:23,sampl:[13,16,20],sample_n_each:13,sample_z:23,save:[5,6,16],scenario:0,schedul:21,scheduler_nam:21,script:0,search:7,search_modul:27,second:[2,14,19,20],see:[13,21],seed:13,select:[5,6,13,18,20,21,23],select_channel:[16,18],separ:13,sequenc:[6,13,15,16,17,18,21,23,26],sequenti:[9,11],serotini:[1,2,3,5,6,8,10],serotiny_zoo_root:21,serv:3,set:[1,5,15,16,20,21],setup:8,sever:6,shall:18,shape:[3,16,20],share:6,should:[3,12,20],show:4,shuffl:15,shuffle_imag:16,sigma1:20,sigma2:20,signatur:3,signifi:2,similar:5,simpl:18,simpli:[3,6,11],singl:[0,15],situat:0,size:[3,12,15,16,20,26],skip_if_exist:[5,6],slower:17,smaller:[13,15],snakemak:3,so:20,some:[0,5,27],someon:3,sort:20,sourc:[11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],spatial_pyramid_pool:[11,24],spatialpyramidpool:26,special:[3,6],specif:[15,18],specifi:[0,3,5,12,21],spheric:5,spirit:5,split:[2,13,15,20],split_column:15,split_datafram:[2,13],split_numb:14,sqrt:20,stabl:[20,21],start:[2,5,6,13,17,18],startswith:[13,17,18],state:21,statist:20,step:[2,5,6],still:0,store:[5,6,16,17,21],store_metadata:21,store_model:21,str:[12,13,15,16,17,18,19,21,23,27],straightforward:0,stratifi:13,strictli:20,string:[0,13,17,18,21,23,26,27],structur:[5,6],sub:12,subclass:[18,23],submit:1,submodul:[9,11],subpackag:9,substr:[13,17,18],sum:[19,23],suppli:[13,17,18],support:[15,17,18,21,23],suppress:16,supress:16,sure:1,sutherland:20,swap:[11,12],swapax:14,symmetr:26,t:[0,8,16,17,21],tabular:23,tabular_conditional_va:[11,21],tabular_condprior_va:[11,21],tabular_va:[11,21],tabularconditionalpriorva:[21,23],tabularconditionalva:[21,23],tabularva:[21,23],tag:1,take:[0,3,17,21],tarbal:8,target:19,target_dim:14,task:5,tensor:[12,16,19,20,21,23],tensor_in:12,term:[21,23],termin:8,test:[1,13,15,20],test_dataload:15,test_epoch_end:21,test_image_output:21,test_pr:21,test_prob:21,test_step:[21,23],test_step_end:21,than:[13,16],the_input_image_col:5,thei:1,them:[2,18],thi:[0,1,2,3,5,6,8,11,13,16,17,18,21,26],third:2,those:6,three:[13,15],through:[1,8],thrown:17,thu:3,ti:4,tiff:[1,16,18],tiff_writ:16,tile_predict:12,to_tensor:14,todo:[4,13],togeth:4,toggl:13,tolstikhin:20,too:18,tool:[2,6],torch:[12,15,16,17,19,20,21,23,24,26],tr:20,track:20,train:[0,2,13,15,20,21,23,24,26],train_dataload:15,train_frac:[2,13],trainer:[10,21,22],training_epoch_end:21,training_step:[21,23],training_step_end:21,transform:[0,5,11,12,16,18],tupl:21,tweak:3,two:[0,19,20],type:[5,6,12,16,17,20,21,23],underli:[0,6],unet:[3,9,11],unetmodel:21,unfinish:[5,6],union:[12,13,15,16,17,20,21,23,26],uniqu:[13,21],unnecessari:0,unpack:6,up:[1,6,21],upon:18,upsampl:13,us:[0,2,3,5,6,11,13,15,16,17,18,20,21,23],usag:24,use_amp:[21,23],user:[2,3,21],usual:5,util:[0,5,6,9,11,15,16,17,18],vae:[11,21],val:[2,13,15],val_dataload:15,val_frac:13,valid:[13,15],validation_epoch_end:21,validation_step:[21,23],validation_step_end:21,valu:[5,6,13,14,18,20,21,23],valueerror:17,variabl:[3,20,21,27],varianc:[19,21,23],variat:[21,23],vector:[15,18],verbos:[5,6,20],veri:6,version:[1,11,20,21],version_str:21,via:[15,18],virtualenv:1,wae:20,wai:3,want:[0,2,21],warn:16,we:[2,3,5,6,17],websit:1,weight:[13,21,23],weight_init:[9,11],welcom:1,what:[5,6,26],when:[1,3,15,16,20],where:[5,6,13,16],wherev:0,whether:[5,6,13,15,16,18,21,23,26],which:[0,2,3,5,6,13,15,17,18,20,21,23],whole:6,wi:20,width:20,wise:19,within:21,without:20,work:[0,1,19],worker:[5,15],workflow:[3,6],wrangl:[0,13],wrap:17,write:16,write_every_n_row:[5,6],x1:24,x2:24,x:[20,21,23,24,26],x_1:20,x_2:20,x_dim:[15,21,23],x_hat:23,x_label:[15,21,23],y:[15,21],y_dim:15,y_encoded_label:18,y_label:[15,21],yaml:[0,3,5,6,10,21,24],yield:3,you:[0,1,2,5,8],your:[0,1,6,8],your_development_typ:1,your_name_her:1,zarr:18,zoo:[9,10,11],zoo_root:21},titles:["serotiny CLI","Contributing","Dataframe wrangling","Dynamic imports","Example workflows","Extraction features from images","Applying image transforms","serotiny","Installation","serotiny","Quickstart","serotiny package","serotiny.data package","serotiny.data.dataframe package","serotiny.data.image package","serotiny.datamodules package","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.loaders package","serotiny.losses package","serotiny.metrics package","serotiny.models package","serotiny.models.callbacks package","serotiny.models.vae package","serotiny.networks package","serotiny.networks.classification package","serotiny.networks.layers package","serotiny.utils package"],titleterms:{"class":18,"import":3,abstract_load:18,activ:26,align:14,appli:6,base_va:23,bind:3,buffered_patch_dataset:16,calculate_blur:20,calculate_fid:20,call:2,callback:22,chain:2,classif:[21,25],cli:0,column:18,comment:6,conditional_va:23,condprior_va:23,config:0,configur:[5,6],content:[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],continuous_bernoulli:19,contribut:1,data:[12,13,14],datafram:[2,13,17,18],dataframe_dataset:17,datamodul:15,deploi:1,dict:0,dummi:15,dynam:3,dynamic_import:27,exampl:[4,5,6],extract:5,featur:5,feature_extract:14,features_to_extract:5,from:[5,8],gaussian:15,get:1,imag:[5,6,14,16],image2d:18,image3d:18,image_va:23,incept:20,indic:7,init:3,instal:8,invok:3,io:[16,17,18],kl_diverg:19,layer:26,load:10,loader:18,loss:19,manifest_datamodul:15,metric:20,mlp:24,model:[10,21,22,23],modul:[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],multipl:2,network:[24,25,26],normal:14,output:6,packag:[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],pad:[14,26],pass:0,patch:15,pearson:20,pipelin:[2,6],plot_xhat:22,positional_arg:3,predict:12,project:14,quickstart:10,quilt:16,reader:17,regress:21,releas:8,resiz:14,sequenti:24,serotini:[0,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],singl:2,sourc:8,spatial_pyramid_pool:26,specifi:6,stabl:8,start:1,submodul:[12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],subpackag:[11,12,16,17,21,24],swap:14,tabl:7,tabular_conditional_va:23,tabular_condprior_va:23,tabular_va:23,test:2,train:10,transform:[2,6,13],transforms_to_appli:6,unet:21,util:[21,27],vae:23,weight_init:24,workflow:4,wrangl:2,zoo:21}})