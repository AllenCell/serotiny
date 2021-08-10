Search.setIndex({docnames:["contributing","index","installation","modules","serotiny","serotiny.datamodules","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.loaders","serotiny.losses","serotiny.metrics","serotiny.models","serotiny.models.callbacks","serotiny.models.deprecated","serotiny.models.vae","serotiny.networks","serotiny.networks.classification","serotiny.networks.deprecated","serotiny.networks.layers","serotiny.networks.vae","serotiny.transform","serotiny.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["contributing.rst","index.rst","installation.rst","modules.rst","serotiny.rst","serotiny.datamodules.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.loaders.rst","serotiny.losses.rst","serotiny.metrics.rst","serotiny.models.rst","serotiny.models.callbacks.rst","serotiny.models.deprecated.rst","serotiny.models.vae.rst","serotiny.networks.rst","serotiny.networks.classification.rst","serotiny.networks.deprecated.rst","serotiny.networks.layers.rst","serotiny.networks.vae.rst","serotiny.transform.rst","serotiny.utils.rst"],objects:{"":{serotiny:[4,0,0,"-"]},"serotiny.datamodules":{DummyDatamodule:[5,1,1,""],ManifestDatamodule:[5,1,1,""],PatchDatamodule:[5,1,1,""],SplitDatamodule:[5,1,1,""],dummy:[5,0,0,"-"],gaussian:[5,0,0,"-"],manifest_datamodule:[5,0,0,"-"],patch:[5,0,0,"-"],split:[5,0,0,"-"]},"serotiny.datamodules.DummyDatamodule":{test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.ManifestDatamodule":{make_dataloader:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.PatchDatamodule":{load_patch_manifest:[5,2,1,""],make_dataloader:[5,2,1,""],make_patch_dataset:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.SplitDatamodule":{generate_args:[5,2,1,""],make_dataloader:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.dummy":{DummyDatamodule:[5,1,1,""],DummyDataset:[5,1,1,""],make_dataloader:[5,3,1,""]},"serotiny.datamodules.dummy.DummyDatamodule":{test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.gaussian":{GaussianDataModule:[5,1,1,""],GaussianDataset:[5,1,1,""],make_dataloader:[5,3,1,""]},"serotiny.datamodules.gaussian.GaussianDataModule":{test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.gaussian.GaussianDataset":{random_corr_mat:[5,2,1,""]},"serotiny.datamodules.manifest_datamodule":{ManifestDatamodule:[5,1,1,""],make_manifest_dataset:[5,3,1,""]},"serotiny.datamodules.manifest_datamodule.ManifestDatamodule":{make_dataloader:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.patch":{PatchDatamodule:[5,1,1,""],make_manifest_dataset:[5,3,1,""]},"serotiny.datamodules.patch.PatchDatamodule":{load_patch_manifest:[5,2,1,""],make_dataloader:[5,2,1,""],make_patch_dataset:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.datamodules.split":{SplitDatamodule:[5,1,1,""],make_dataloader:[5,3,1,""]},"serotiny.datamodules.split.SplitDatamodule":{generate_args:[5,2,1,""],make_dataloader:[5,2,1,""],test_dataloader:[5,2,1,""],train_dataloader:[5,2,1,""],val_dataloader:[5,2,1,""]},"serotiny.io":{buffered_patch_dataset:[6,0,0,"-"],dataframe:[7,0,0,"-"],image:[6,0,0,"-"],transforms:[6,0,0,"-"],utils:[6,0,0,"-"]},"serotiny.io.buffered_patch_dataset":{BufferedPatchDataset:[6,1,1,""]},"serotiny.io.buffered_patch_dataset.BufferedPatchDataset":{get_buffer_history:[6,2,1,""],get_patch:[6,2,1,""],get_random_patch:[6,2,1,""],insert_new_element_into_buffer:[6,2,1,""]},"serotiny.io.dataframe":{dataframe_dataset:[7,0,0,"-"],loaders:[8,0,0,"-"],split:[7,0,0,"-"],utils:[7,0,0,"-"]},"serotiny.io.dataframe.dataframe_dataset":{DataframeDataset:[7,1,1,""]},"serotiny.io.dataframe.loaders":{Load2DImage:[8,1,1,""],Load3DImage:[8,1,1,""],LoadClass:[8,1,1,""],LoadColumn:[8,1,1,""],LoadColumns:[8,1,1,""],abstract_loader:[8,0,0,"-"],classes:[8,0,0,"-"],columns:[8,0,0,"-"],image2d:[8,0,0,"-"],image3d:[8,0,0,"-"],utils:[8,0,0,"-"]},"serotiny.io.dataframe.loaders.abstract_loader":{Loader:[8,1,1,""]},"serotiny.io.dataframe.loaders.classes":{LoadClass:[8,1,1,""]},"serotiny.io.dataframe.loaders.columns":{LoadColumn:[8,1,1,""],LoadColumns:[8,1,1,""]},"serotiny.io.dataframe.loaders.image2d":{Load2DImage:[8,1,1,""]},"serotiny.io.dataframe.loaders.image3d":{Load3DImage:[8,1,1,""]},"serotiny.io.dataframe.loaders.utils":{load_transforms:[8,3,1,""]},"serotiny.io.dataframe.split":{split_dataset:[7,3,1,""]},"serotiny.io.dataframe.utils":{append_one_hot:[7,3,1,""],filter_columns:[7,3,1,""],load_csv:[7,3,1,""],one_hot_encoding:[7,3,1,""]},"serotiny.io.image":{change_resolution:[6,3,1,""],define_channels:[6,3,1,""],infer_dims:[6,3,1,""],png_loader:[6,3,1,""],project_2d:[6,3,1,""],subset_channels:[6,3,1,""],tiff_loader:[6,3,1,""],tiff_loader_CZYX:[6,3,1,""],tiff_writer:[6,3,1,""]},"serotiny.io.transforms":{CropCenter:[6,1,1,""],MinMaxNormalize:[6,1,1,""],PadTo:[6,1,1,""],Permute:[6,1,1,""],ResizeBy:[6,1,1,""],ResizeTo:[6,1,1,""]},"serotiny.io.utils":{download_quilt_data:[6,3,1,""]},"serotiny.losses":{elbo:[9,0,0,"-"]},"serotiny.losses.elbo":{calculate_elbo:[9,3,1,""],diagonal_gaussian_kl:[9,3,1,""],isotropic_gaussian_kl:[9,3,1,""]},"serotiny.metrics":{InceptionV3:[10,1,1,""],calculate_blur:[10,0,0,"-"],calculate_fid:[10,0,0,"-"],inception:[10,0,0,"-"]},"serotiny.metrics.InceptionV3":{BLOCK_INDEX_BY_DIM:[10,4,1,""],DEFAULT_BLOCK_INDEX:[10,4,1,""],forward:[10,2,1,""],training:[10,4,1,""]},"serotiny.metrics.calculate_blur":{calculate_blur:[10,3,1,""]},"serotiny.metrics.calculate_fid":{calculate_fid:[10,3,1,""],get_activations:[10,3,1,""]},"serotiny.metrics.inception":{InceptionV3:[10,1,1,""]},"serotiny.metrics.inception.InceptionV3":{BLOCK_INDEX_BY_DIM:[10,4,1,""],DEFAULT_BLOCK_INDEX:[10,4,1,""],forward:[10,2,1,""],training:[10,4,1,""]},"serotiny.models":{ClassificationModel:[11,1,1,""],ImageVAE:[11,1,1,""],RegressionModel:[11,1,1,""],TabularConditionalPriorVAE:[11,1,1,""],TabularConditionalVAE:[11,1,1,""],TabularVAE:[11,1,1,""],UnetModel:[11,1,1,""],callbacks:[12,0,0,"-"],classification:[11,0,0,"-"],deprecated:[13,0,0,"-"],regression:[11,0,0,"-"],unet:[11,0,0,"-"],vae:[14,0,0,"-"],zoo:[11,0,0,"-"]},"serotiny.models.ClassificationModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],on_after_backward:[11,2,1,""],parse_batch:[11,2,1,""],precision:[11,4,1,""],test_epoch_end:[11,2,1,""],test_step:[11,2,1,""],test_step_end:[11,2,1,""],training:[11,4,1,""],training_step:[11,2,1,""],training_step_end:[11,2,1,""],use_amp:[11,4,1,""],validation_epoch_end:[11,2,1,""],validation_step:[11,2,1,""],validation_step_end:[11,2,1,""]},"serotiny.models.ImageVAE":{training:[11,4,1,""]},"serotiny.models.RegressionModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],parse_batch:[11,2,1,""],precision:[11,4,1,""],test_epoch_end:[11,2,1,""],test_step:[11,2,1,""],training:[11,4,1,""],training_epoch_end:[11,2,1,""],training_step:[11,2,1,""],use_amp:[11,4,1,""],validation_epoch_end:[11,2,1,""],validation_step:[11,2,1,""]},"serotiny.models.TabularConditionalPriorVAE":{training:[11,4,1,""]},"serotiny.models.TabularConditionalVAE":{parse_batch:[11,2,1,""],training:[11,4,1,""]},"serotiny.models.TabularVAE":{training:[11,4,1,""]},"serotiny.models.UnetModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],get_unet_padding:[11,2,1,""],on_after_backward:[11,2,1,""],parse_batch:[11,2,1,""],precision:[11,4,1,""],test_step:[11,2,1,""],test_step_end:[11,2,1,""],training:[11,4,1,""],training_step:[11,2,1,""],use_amp:[11,4,1,""],validation_step:[11,2,1,""]},"serotiny.models.classification":{ClassificationModel:[11,1,1,""],acc_prec_recall:[11,3,1,""]},"serotiny.models.classification.ClassificationModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],on_after_backward:[11,2,1,""],parse_batch:[11,2,1,""],test_epoch_end:[11,2,1,""],test_step:[11,2,1,""],test_step_end:[11,2,1,""],training:[11,4,1,""],training_step:[11,2,1,""],training_step_end:[11,2,1,""],validation_epoch_end:[11,2,1,""],validation_step:[11,2,1,""],validation_step_end:[11,2,1,""]},"serotiny.models.deprecated":{CBVAEMLPModel:[13,1,1,""],cbvae_mlp:[13,0,0,"-"]},"serotiny.models.deprecated.CBVAEMLPModel":{configure_optimizers:[13,2,1,""],forward:[13,2,1,""],parse_batch:[13,2,1,""],precision:[13,4,1,""],test_epoch_end:[13,2,1,""],test_step:[13,2,1,""],training:[13,4,1,""],training_step:[13,2,1,""],use_amp:[13,4,1,""],validation_step:[13,2,1,""]},"serotiny.models.deprecated.cbvae_mlp":{CBVAEMLPModel:[13,1,1,""]},"serotiny.models.deprecated.cbvae_mlp.CBVAEMLPModel":{configure_optimizers:[13,2,1,""],forward:[13,2,1,""],parse_batch:[13,2,1,""],test_epoch_end:[13,2,1,""],test_step:[13,2,1,""],training:[13,4,1,""],training_step:[13,2,1,""],validation_step:[13,2,1,""]},"serotiny.models.regression":{RegressionModel:[11,1,1,""]},"serotiny.models.regression.RegressionModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],parse_batch:[11,2,1,""],test_epoch_end:[11,2,1,""],test_step:[11,2,1,""],training:[11,4,1,""],training_epoch_end:[11,2,1,""],training_step:[11,2,1,""],validation_epoch_end:[11,2,1,""],validation_step:[11,2,1,""]},"serotiny.models.unet":{UnetModel:[11,1,1,""]},"serotiny.models.unet.UnetModel":{configure_optimizers:[11,2,1,""],forward:[11,2,1,""],get_unet_padding:[11,2,1,""],on_after_backward:[11,2,1,""],parse_batch:[11,2,1,""],test_step:[11,2,1,""],test_step_end:[11,2,1,""],training:[11,4,1,""],training_step:[11,2,1,""],validation_step:[11,2,1,""]},"serotiny.models.vae":{base_vae:[14,0,0,"-"],conditional_vae:[14,0,0,"-"],condprior_vae:[14,0,0,"-"],image_vae:[14,0,0,"-"],tabular_conditional_vae:[14,0,0,"-"],tabular_condprior_vae:[14,0,0,"-"],tabular_vae:[14,0,0,"-"]},"serotiny.models.vae.base_vae":{BaseVAE:[14,1,1,""]},"serotiny.models.vae.base_vae.BaseVAE":{configure_optimizers:[14,2,1,""],forward:[14,2,1,""],parse_batch:[14,2,1,""],sample_z:[14,2,1,""],test_step:[14,2,1,""],training:[14,4,1,""],training_step:[14,2,1,""],validation_step:[14,2,1,""]},"serotiny.models.vae.conditional_vae":{ConditionalVAE:[14,1,1,""]},"serotiny.models.vae.conditional_vae.ConditionalVAE":{parse_batch:[14,2,1,""],precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.vae.condprior_vae":{ConditionalPriorVAE:[14,1,1,""]},"serotiny.models.vae.condprior_vae.ConditionalPriorVAE":{forward:[14,2,1,""],precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.vae.image_vae":{ImageVAE:[14,1,1,""]},"serotiny.models.vae.image_vae.ImageVAE":{precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.vae.tabular_conditional_vae":{TabularConditionalVAE:[14,1,1,""]},"serotiny.models.vae.tabular_conditional_vae.TabularConditionalVAE":{parse_batch:[14,2,1,""],precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.vae.tabular_condprior_vae":{TabularConditionalPriorVAE:[14,1,1,""]},"serotiny.models.vae.tabular_condprior_vae.TabularConditionalPriorVAE":{precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.vae.tabular_vae":{TabularVAE:[14,1,1,""]},"serotiny.models.vae.tabular_vae.TabularVAE":{precision:[14,4,1,""],training:[14,4,1,""],use_amp:[14,4,1,""]},"serotiny.models.zoo":{build_model_path:[11,3,1,""],get_checkpoint_callback:[11,3,1,""],get_model:[11,3,1,""],get_root:[11,3,1,""],get_trainer_at_checkpoint:[11,3,1,""],store_metadata:[11,3,1,""],store_model:[11,3,1,""]},"serotiny.networks":{deprecated:[17,0,0,"-"],layers:[18,0,0,"-"],mlp:[15,0,0,"-"],sequential:[15,0,0,"-"],vae:[19,0,0,"-"],weight_init:[15,0,0,"-"]},"serotiny.networks.deprecated":{CBVAEDecoder:[17,1,1,""],CBVAEDecoderMLP:[17,1,1,""],CBVAEEncoder:[17,1,1,""],CBVAEEncoderMLP:[17,1,1,""],cbvae_decoder:[17,0,0,"-"],cbvae_decoder_mlp:[17,0,0,"-"],cbvae_encoder:[17,0,0,"-"],cbvae_encoder_mlp:[17,0,0,"-"]},"serotiny.networks.deprecated.CBVAEDecoder":{forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.CBVAEDecoderMLP":{decoder:[17,2,1,""],forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.CBVAEEncoder":{conv_forward:[17,2,1,""],forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.CBVAEEncoderMLP":{encoder:[17,2,1,""],forward:[17,2,1,""],sampling:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.cbvae_decoder":{CBVAEDecoder:[17,1,1,""]},"serotiny.networks.deprecated.cbvae_decoder.CBVAEDecoder":{forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.cbvae_decoder_mlp":{CBVAEDecoderMLP:[17,1,1,""]},"serotiny.networks.deprecated.cbvae_decoder_mlp.CBVAEDecoderMLP":{decoder:[17,2,1,""],forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.cbvae_encoder":{CBVAEEncoder:[17,1,1,""]},"serotiny.networks.deprecated.cbvae_encoder.CBVAEEncoder":{conv_forward:[17,2,1,""],forward:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.deprecated.cbvae_encoder_mlp":{CBVAEEncoderMLP:[17,1,1,""]},"serotiny.networks.deprecated.cbvae_encoder_mlp.CBVAEEncoderMLP":{encoder:[17,2,1,""],forward:[17,2,1,""],sampling:[17,2,1,""],training:[17,4,1,""]},"serotiny.networks.layers":{activation:[18,0,0,"-"],pad:[18,0,0,"-"],spatial_pyramid_pool:[18,0,0,"-"]},"serotiny.networks.layers.activation":{activation_map:[18,3,1,""]},"serotiny.networks.layers.pad":{PadLayer:[18,1,1,""]},"serotiny.networks.layers.pad.PadLayer":{forward:[18,2,1,""],training:[18,4,1,""]},"serotiny.networks.layers.spatial_pyramid_pool":{SpatialPyramidPool:[18,1,1,""],spatial_pyramid_pool:[18,3,1,""]},"serotiny.networks.layers.spatial_pyramid_pool.SpatialPyramidPool":{forward:[18,2,1,""],training:[18,4,1,""]},"serotiny.networks.mlp":{MLP:[15,1,1,""]},"serotiny.networks.mlp.MLP":{forward:[15,2,1,""],training:[15,4,1,""]},"serotiny.networks.sequential":{Sequential:[15,1,1,""]},"serotiny.networks.sequential.Sequential":{forward:[15,2,1,""],training:[15,4,1,""]},"serotiny.networks.vae":{CBVAEDecoderMLP:[19,1,1,""],CBVAEEncoderMLP:[19,1,1,""],cbvae_decoder_mlp:[19,0,0,"-"],cbvae_encoder_conv:[19,0,0,"-"],cbvae_encoder_mlp:[19,0,0,"-"]},"serotiny.networks.vae.CBVAEDecoderMLP":{decoder:[19,2,1,""],forward:[19,2,1,""],training:[19,4,1,""]},"serotiny.networks.vae.CBVAEEncoderMLP":{encoder:[19,2,1,""],forward:[19,2,1,""],sampling:[19,2,1,""],training:[19,4,1,""]},"serotiny.networks.vae.cbvae_decoder_mlp":{CBVAEDecoderMLP:[19,1,1,""]},"serotiny.networks.vae.cbvae_decoder_mlp.CBVAEDecoderMLP":{decoder:[19,2,1,""],forward:[19,2,1,""],training:[19,4,1,""]},"serotiny.networks.vae.cbvae_encoder_conv":{CBVAEEncoderMLP:[19,1,1,""]},"serotiny.networks.vae.cbvae_encoder_conv.CBVAEEncoderMLP":{encoder:[19,2,1,""],forward:[19,2,1,""],sampling:[19,2,1,""],training:[19,4,1,""]},"serotiny.networks.vae.cbvae_encoder_mlp":{CBVAEEncoderMLP:[19,1,1,""]},"serotiny.networks.vae.cbvae_encoder_mlp.CBVAEEncoderMLP":{encoder:[19,2,1,""],forward:[19,2,1,""],sampling:[19,2,1,""],training:[19,4,1,""]},"serotiny.networks.weight_init":{weight_init:[15,3,1,""]},"serotiny.transform":{pad:[20,0,0,"-"],swap:[20,0,0,"-"]},"serotiny.transform.pad":{ExpandColumns:[20,1,1,""],ExpandTo:[20,1,1,""],expand_columns:[20,3,1,""],expand_to:[20,3,1,""],pull_to:[20,3,1,""],split_number:[20,3,1,""],to_tensor:[20,3,1,""]},"serotiny.transform.swap":{SwapAxes:[20,1,1,""]},"serotiny.utils":{dynamic_imports:[21,0,0,"-"]},"serotiny.utils.dynamic_imports":{bind:[21,3,1,""],get_name_and_arguments:[21,3,1,""],get_name_from_path:[21,3,1,""],init:[21,3,1,""],invoke:[21,3,1,""],load_config:[21,3,1,""],load_multiple:[21,3,1,""],module_get:[21,3,1,""],module_or_path:[21,3,1,""],module_path:[21,3,1,""],search_modules:[21,3,1,""]},serotiny:{datamodules:[5,0,0,"-"],get_module_version:[4,3,1,""],io:[6,0,0,"-"],losses:[9,0,0,"-"],metrics:[10,0,0,"-"],models:[11,0,0,"-"],networks:[15,0,0,"-"],transform:[20,0,0,"-"],utils:[21,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[5,6,10,11,13,14],"00028":[11,13,14],"001":[11,13,14],"01":[11,13,14],"02":[11,13,14],"06":10,"1":[1,5,6,8,10,11,13,14],"10":[5,11,13,14],"16":[11,13,14],"1704":[11,13,14],"192":10,"1e":[10,11,13,14],"2":[1,6,10,11,13,14],"2048":10,"25":[11,17,19],"256":[15,17,19],"295":[17,19],"299":10,"2d":[1,6,11,14,17,18],"3":[6,10,11,13,14],"32":6,"344":10,"3d":[1,6,17,18],"5":[5,11,13,14],"6":[11,13,14],"64":[6,10,17,19],"768":10,"99":[11,13,14],"boolean":[11,14],"case":[5,11,13,14],"class":[1,5,6,7,9,10,11,13,14,15,17,18,19,20,21],"default":[11,13,14],"do":[5,11,13,14],"final":[10,17],"float":[5,6,8,11,14],"function":[1,7,10,11,13,14,15,17,18,19,21],"import":1,"int":[5,6,10,11,13,14,17,18,20],"new":[0,6],"public":2,"return":[4,5,6,7,10,11,13,14],"short":0,"switch":11,"true":[5,6,7,10,11,13,14],"while":[11,13,15,17,18,19],A:[0,1,5,7,10,11,13,14],As:10,At:[11,13,14],But:[11,13,14],For:[1,5],If:[2,5,6,10,11,13,14],In:[5,7,11,13,14],It:[0,5,11,13,14],One:7,Or:2,The:[1,2,5,6,7,10,11,13,14],Then:0,There:[5,11,13,14],To:2,With:[11,13],_:1,_get_checkpoint:1,_loss:[9,11,14],_xd:17,ab:[11,13,14],abl:10,abov:[5,11,13,14],abstract_load:[6,7],acc:[11,13,14],acc_prec_recal:11,accord:[17,18],accuraci:[11,13,14],across:6,act:1,activ:[4,10,15,17],activation_last:17,activation_map:18,actual:[11,13,14],adam:[11,13,14],add:[0,5,11,13,14],add_histogram:11,add_imag:[11,13,14],addit:[11,13,14],advanc:11,after:[11,13,14],afterward:[15,17,18,19],agglomer:4,aic:[4,6,7],aicsimag:6,algorithm:[11,13,14],all:[0,1,5,6,10,11,13,14,15,17,18,19],all_test_pr:[11,13],all_test_step_out:11,allencel:6,allencellmodel:[1,2],allow:15,along:[11,13,14],alreadi:5,also:[0,5,11,13,14,17],although:[15,17,18,19],alwai:[0,2],amp:[11,13,14],an:[1,4,5,6,7,10,11,13,14],anaconda:0,ani:[5,11,13,14],anisotrop:[11,14],anyth:[11,13,14],append:[7,11,13,14],append_one_hot:7,appli:[1,6,11,15,17,18],appreci:0,appropri:7,ar:[0,1,5,7,10,11,13,14],arbitrari:[5,10],arg:[5,11,13,14],argmax:[11,13,14],argument:[5,6,7,11,13,14],arrai:[1,7,10,11,14,20],arraylik:6,arxiv:[11,13,14],ascend:10,assembl:1,assign:5,associ:[1,11,13,14],assum:5,attrib:6,attribut:6,auto_pad:11,autoencod:[11,13,14],autograd:10,automat:[11,13,14],aux:10,auxiliari:15,avail:11,averag:[10,11,13,14],ax:20,axi:6,b:0,back:[7,11,13,14],backend:11,backprop:[11,13,14],backward:[11,13,14],bar:[11,13,14],base:[1,5,6,7,8,10,11,13,14,15,17,18,19,20],base_va:[4,11],baseva:[11,14],basic:[11,14],batch:[5,10,11,13,14],batch_cifar:5,batch_idx:[11,13,14],batch_mnist:5,batch_parts_output:11,batch_siz:[5,10],becaus:11,been:[5,11,13,14],befor:[6,10,11,17],being:[11,13,14,17],below:[11,13,14],beta:[5,9,11,13,14],better:5,between:[6,10,11,13,14],big:11,bilinearli:10,bimod:5,binari:8,bind:21,binomi:5,bit:[0,11,13,14],blob:10,block:10,block_index_by_dim:10,blurri:10,bool:[5,6,10,11,13,14,15,17,18,19],both:[6,15,17,18,19],br:0,branch:0,bucket:6,buffer:6,buffer_index:6,buffer_s:6,buffer_switch_interv:6,buffered_patch_dataset:[3,4],bufferedpatchdataset:6,bugfix:0,build:[0,10],build_model_path:11,built:1,bump2vers:0,bx3xhxw:10,c:[17,19],c_1:10,c_2:10,c_dim:[11,14,17,19],c_label:[5,11,13,14],c_label_ind:[5,13],calc_all_result:[11,13],calcul:[10,11,13,14],calculate_blur:[3,4],calculate_elbo:9,calculate_fid:[3,4],call:[5,7,11,13,14,15,17,18,19],callabl:[5,6,11,14],callback:[4,11],can:[0,1,2,5,11,13,14],care:[15,17,18,19],cbvae:[4,11],cbvae_decod:[4,15],cbvae_decoder_mlp:[4,15],cbvae_encod:[4,15],cbvae_encoder_conv:[4,15],cbvae_encoder_mlp:[4,15],cbvae_mlp:[4,11],cbvaedecod:17,cbvaedecodermlp:[17,19],cbvaeencod:17,cbvaeencodermlp:[17,19],cbvaemlpmodel:13,cd:0,cell:[4,7],cell_coeff:13,center_of_mass:6,chang:[0,1,6,11],change_resolut:6,channel:[5,6,11,14,17],channel_fan:11,channel_index:[6,8],channel_mask:6,channel_ord:6,channel_subset:6,check:[0,7],checkout:0,checkpoint_mod:11,checkpoint_monitor:11,choos:[11,13,14],cifar:5,cifar_load:5,ckpt_path:1,classif:[3,4,15],classifi:[10,11],classificationmodel:11,classnam:21,clean:1,clip_max:6,clip_min:6,clone:[0,2],closur:[11,13,14],code:[1,11],collect:5,cols_to_filt:7,column:[1,5,6,7,11,20],com:[0,1,2,10],command:[1,2,11],commit:0,commonli:1,complet:4,compos:5,composit:15,comput:[11,13,14,15,17,18,19],condit:[5,11,13,14,17],condition_mod:[11,14],conditional_va:[4,11],conditionalpriorva:[11,14],conditionalva:[11,14],condprior_va:[4,11],config:[15,21],configur:[11,13,14],configure_optim:[11,13,14],connect:10,consist:7,constant:6,construct:1,contain:[6,7,8,10,11,13,14],content:3,continu:[11,13,14],contribut:1,control:[11,13,14],conv:[11,17],conv_channels_list:17,conv_forward:17,convert:7,convolut:[10,17],copi:2,core:[5,7,11,13,14],corr:5,correct:5,correl:5,correspond:[6,7,10,11,13,14,17],cosineann:[11,13,14],could:[11,13,14],covari:[10,11,14],crash:[11,13,14],creat:[0,5],create_datamodul:1,credit:0,cropcent:6,cropi:6,cropx:6,cropz:6,cuda:10,curl:2,current:[11,13,14],custom:[11,13,14],cycl:[11,13,14],d:[5,10,11,13,14],data:[1,4,5,6,7,10,11,13,14,17,18],data_new:6,data_save_loc:6,datafram:[4,6],dataframe_dataset:[4,6],dataframedataset:7,dataload:[5,11,13,14],dataloader_i_output:11,dataloader_idx:[5,11,13,14],dataloader_out:11,dataloader_output:[11,13],dataloader_output_result:11,datamodul:[3,4],datamodule_config:1,datamodule_nam:1,dataset:[5,6,7,11,13,14],ddp2:11,ddp:11,decid:[11,13,14],decod:[11,13,14,17,19],deep:1,def:[5,11,13,14],default_block_index:10,defin:[11,13,14,15,17,18,19],define_channel:6,denomintaor:11,depend:[5,10],deprec:[4,11,15],depth:11,describ:[11,13,14,17],descript:0,desir:6,detail:[0,11],determin:[6,11,14],dev:0,develop:0,diagon:[5,11,14],diagonal_gaussian_kl:9,dict:[5,6,7,11,13,14,15],dictionari:[5,6,11,13,14],didn:[11,13],differ:[11,13,14],dim:[5,10,11,13,14,15],dimens:[5,6,10,11,17,18,20],dimension:[10,17,18],directli:8,dis_opt:[11,13,14],dis_sch:[11,13,14],disabl:[6,11,13,14],displai:[11,13,14],distanc:10,distribut:[5,11],don:[2,5,11,13,14],done:0,doubl:11,dougal:10,down:[1,11],down_residu:17,download:[2,5,6],download_quilt_data:6,downresiduallay:17,dp:11,drop:5,drop_last:5,dtype:[6,8],dummi:[3,4],dummydatamodul:5,dummydataset:5,dynamic_import:[3,4],e:[0,11,13,14,21],each:[5,7,10,11,13,14,18],easili:1,edit:0,either:[2,5,7,11,14],elbo:[3,4],element:6,enabl:[11,13,14],encod:[1,7,11,13,14,17,19],end:[11,13,14],endswith:[7,8],ensur:11,entri:[11,13],environ:0,ep:10,epoch:[5,11,13,14],essenti:1,eval:[11,13,14],everi:[0,5,11,13,14,15,17,18,19],everyth:5,ex:0,exampl:[5,6,11,13,14],example_imag:[11,13,14],exchang:6,exclud:[7,8],exist:7,expand_column:20,expand_to:20,expandcolumn:20,expanded_column:20,expandto:20,expect:[10,11,14],experi:[11,13,14],explain:6,exponentiallr:[11,13,14],extract:6,factor:6,fals:[1,5,6,8,10,11,13,14],fancier:[11,13,14],featur:[0,10],fed:[10,17],feed:10,field:[7,8],file:[0,5,6,7,11],filter:1,filter_column:7,final_metr:[11,13],final_valu:[11,13],finetun:10,fire:1,first:[10,11,13,14,20],fit:[5,11,13,14],fix:[11,14],flag:[6,11,14],float32:6,float64:6,fm:5,folder:5,follow:5,fork:0,form:5,former:[11,13,14,15,17,18,19],forward:[10,11,13,14,15,17,18,19],found:[0,11,13,14],frame:7,frechet:10,frequenc:[11,13,14],from:[5,6,7,8,11,13,14,15,18],full:[1,11],fulli:10,g:[11,13,14,21],gan:[11,13,14],gather:1,gaussian:[3,4,10],gaussiandatamodul:5,gaussiandataset:5,gen_opt:[11,13,14],gen_sch:[11,13,14],gener:[1,5,6,7,10,11,13,14],generate_arg:5,get:[10,11,13,14],get_activ:10,get_buffer_histori:6,get_checkpoint_callback:11,get_image_data:6,get_model:[1,11],get_module_vers:4,get_name_and_argu:21,get_name_from_path:21,get_patch:6,get_predict:10,get_random_patch:6,get_root:11,get_trainer_at_checkpoint:[1,11],get_unet_pad:11,gh:0,git:[0,1,2],github:[0,1,2,10],give:5,given:[0,5,6,7,8,10,11,13,14,17,18,21],global_step:11,goe:[11,13,14],gpu:[5,10,11,13,14],gpu_0_pr:11,gpu_1_pr:11,gpu_n_pr:11,grad:11,gradient:[10,11,13,14],greatli:0,grid:[11,13,14],guid:[2,11],ha:[5,6,11,13,14],handl:[0,5,10,11,13,14],happen:5,hardwar:[5,10],have:[2,5,11,13,14],head:1,height:10,help:0,here:[0,5,11,13,14],hi:10,hidden:[11,13,14],hidden_channel:[11,14],hidden_lay:[11,14,15,17,19],hipsc_single_cell_image_dataset:6,hook:[11,13,14,15,17,18,19],hot:[1,7],how:[0,11,13,14,17],howev:[5,11],html:0,http:[1,2,10,11,13,14],huge:11,i:[11,14],ideal:11,ignor:[15,17,18,19],ignore_warn:6,iloc:[5,7],imag:[1,3,4,5,7,8,10,11,13,14,17],image2d:[6,7],image3d:[6,7],image_va:[4,11],imageva:[11,14],img:6,implement:[5,10,11,13,14],improv:[11,13,14],imsize_compress:17,in_channel:[11,14],incept:[3,4],inceptionv3:10,includ:[0,1,11,13,14],index:[1,6,7,8,10,11,13,14],indic:[5,6,10,11],individu:[11,13],infer_dim:6,inform:[1,11],init:[1,21],initi:[11,14,15,17,18,19],inner:[11,13,17],inp:10,input:[1,5,6,7,10,11,14,17],input_channel:11,input_dim:[11,14,17],insert:6,insert_new_element_into_buff:6,insid:11,inspect:11,instal:0,instanc:[10,11,13,14,15,17,18,19],instanti:[5,11,14,15,17],instead:[15,17,18,19],integ:[11,13,14],intend:1,interest:[11,13,14],interfac:[1,11,14],intermedi:17,intern:[15,17,18,19],interv:[11,13,14],invok:[1,21],io:[1,3,4],isotrop:[9,11,13,14],isotropic_gaussian_kl:9,item:[6,11,13,14],iter:5,its:[1,7,11,13,14],j:10,k:11,keep:10,kei:[5,6,7,11,13,14,21],keyword:[11,13,14],kld:[11,14],know:[11,13,14],kwarg:[5,11,13,14],label:[1,5,11,14],labels_hat:[11,13,14],larg:11,larger:[1,5],last:[5,6,11,13,14],latent:17,latent_dim:[11,14,17,19],later:11,latter:[11,13,14,15,17,18,19],layer:[4,10,15,17],lbfg:[11,13,14],learn:[1,11,13,14],learn_prior_logvar:[11,13,14],learningratemonitor:[11,13,14],least:[11,14],len:[11,13,14],length:[5,6],let:6,leverag:17,librari:1,licens:1,lie:10,lightn:[1,5,11,13,14],lightningdatamodul:5,lightningmodul:[11,13,14],like:[10,11,13,14],line:[10,11],lint:0,list:[5,6,10,11,13,14],littl:0,load2dimag:8,load3dimag:8,load:[5,6,7,11],load_config:21,load_csv:7,load_multipl:21,load_patch_manifest:5,load_transform:8,loadclass:8,loadcolumn:8,loader:[5,6,7],loader_a:5,loader_b:5,loader_dict:5,loader_n:5,local:0,log:[11,13,14],log_dict:[11,13,14],log_var:[9,14,17,19],logger:[11,13,14],logic:[5,11],logvar1:9,logvar2:9,loop:[10,11],loss:[3,4,11,13,14],lr:[11,13,14],lr_dict:[11,13,14],lr_schedul:[11,13,14],lstm:[11,13,14],m0:11,m1:11,m2:11,m2r:0,m3:11,m4:11,m5:11,m6:11,m7:11,m:[0,15],machin:1,main:2,maintain:0,major:0,make:[0,7,11],make_dataload:5,make_grid:[11,13,14],make_manifest_dataset:5,make_patch_dataset:5,mani:17,manifest:[5,7],manifest_datamodul:[3,4],manifestdatamodul:5,map:[6,10,18],mask:[5,6,13],mask_thresh:6,master:10,match:[5,11],matrix:[5,10,11,14],max:10,md:1,mean:[9,10],mention:[11,13,14],merg:[5,7],metadata:11,method:[2,5,6,7,11,13,14],metric:[3,4,11,13,14],might:[10,11,13,14],min:10,minmaxnorm:6,minor:0,mit:1,mitotic_class:11,ml:4,mlp:[3,4],mnist:5,mnist_load:5,mode:[0,5,6,9,11,13,14],model:[3,4,10,15,17],model_class:11,model_di:[11,13,14],model_gen:[11,13,14],model_id:11,model_path:[1,11],model_root:11,model_zoo_path:1,modifi:7,modul:[1,3],modular:1,module_get:[1,21],module_or_path:21,module_path:21,moduledict:11,monitor:[11,13,14],more:[5,6,11],most:[2,11,13,14],mseloss:[9,11,14],mu1:[9,10],mu2:[9,10],mu:[14,17,19],mu_1:10,mu_2:10,multi:[11,13,14],multi_gpu:11,multipl:[5,11,13,14],multivari:10,must:[10,11,13,14],n:[1,10,11,20],n_ch_ref:17,n_ch_target:17,n_class:[11,17],n_critic:[11,13,14],n_imag:10,n_latent_dim:17,name:[5,6,11,13,14],named_paramet:11,nce:11,nce_loss:11,nd:6,ndarrai:6,nearest:6,necessari:[5,6],need:[5,10,11,13,14,15,17,18,19],nest:5,net:[10,17],network:[1,3,4,10,11,14],network_config:11,neural:11,next:[11,13,14],nf:6,nn:[9,10,11,13,14,15,17,18,19],none:[5,6,7,8,9,11,13,14,15,17,20],normal:[5,7,10,11,13,14],normalize_input:10,note:[5,11,13,14],now:[0,11],np:[6,7],num:10,num_channel:[6,8],num_class:8,num_gpu:11,num_work:[1,5],number:[5,6,10,11,13,14,17],numpi:[6,10,11,14],object:[6,8,11,13,14,20],off:[5,6],often:[11,13,14],ol:2,om:6,on_after_backward:11,onc:2,one:[1,5,7,11,13,14,15,17,18,19],one_hot:7,one_hot_encod:7,onli:[5,11,13,14],onto:7,oper:[11,13,14],opt:[11,14],optim:[11,13,14],optimizer_idx:[11,13,14],optimizer_on:[11,13,14],optimizer_step:[11,13,14],optimizer_two:[11,13,14],option:[5,6,7,11,13,14,17],order:[5,6,11,13,14],org:[11,13,14],origin:[0,6,7],other:[0,11],out:[1,11,13,14],out_pool_s:18,out_step:10,outer:[11,13],output:[6,10,11,13,14,17],output_block:10,output_channel:11,output_dtyp:6,output_path:11,output_result:11,over:[5,10,11,13,14],overrid:[11,13,14],overridden:[11,15,17,18,19],overwrit:6,own:[11,13,14],packag:[0,1,3],pad:[3,4,6,11,15,17],pad_dim:18,padding_lat:17,padlay:18,padto:6,page:[1,5],panda:7,param:10,paramet:[5,6,7,10,11,13,14,17,18],paramref:[5,11,13,14],parquet:5,parse_batch:[11,13,14],part:11,pass:[0,11,13,14,15,17,18,19],patch:[0,3,4,6],patch_column:6,patch_shap:6,patchdatamodul:5,path:[5,6,7,8,11,17,21],path_2d:6,path_3d:6,path_in:6,path_out:6,path_str:6,pathlib:[5,6,7],pattern:5,pd:7,per:[11,13],perform:[5,15,17,18,19],permut:[5,6],pin_memori:[1,5],pip:[0,1,2],pipelin:1,pixel:6,place:[1,11],pleas:[1,5],png:6,png_loader:6,point:7,pool:10,pool_3:10,pop:1,portion:[11,17],possibl:[0,5,10,11,13,14],practic:7,precalcul:10,precis:[11,13,14],pred:11,predict:[11,13],prefer:2,prepar:5,prepare_data:5,present:[7,11,13,14],pretrain:10,previou:[11,13,14],primit:4,print:11,prior:[11,14],prior_encod:14,prior_encoder_hidden_lay:[11,14],prior_logvar:[9,11,13,14],prior_mod:[11,13,14],prior_mu:9,problem:11,procedur:[11,13,14],process:[1,2,5],produc:[11,13,14],progress:[11,13,14],proj_al:6,project:[0,1,5,6,17],project_2d:6,projection_imag:11,propag:[11,13,14],provid:[1,5,6],pseudocod:[11,13,14],publish:0,pull:0,pull_to:20,push:0,put:[11,13,14],py:[2,10,17],pypi:0,python:[0,2],pytorch:[1,5,7,11,13,14],pytorch_integrated_cel:17,pytorch_lightn:[5,11,13,14],queri:10,quilt:6,random:[5,6],random_corr_mat:5,rang:10,rate:[11,13,14],ratio:7,raw:0,re:0,read:[0,7],readi:0,reason:10,recent:2,recip:[15,17,18,19],recognit:11,recommend:[0,5],recon_loss:[9,11,14],reconstruct:[11,14],reconstructed_x:9,reconstruction_loss:[11,14],reducelronplateau:[11,13,14],refer:17,regex:[7,8],regist:[15,17,18,19],regress:[3,4],regressionmodel:11,relat:1,releas:[0,1],reload_callback:11,reload_dataloaders_every_epoch:5,reload_logg:11,remind:0,repo:[0,2],report:10,repositori:2,represent:10,request:0,requir:[7,10,11,13,14],required_field:7,requires_grad:10,resiz:10,resize_input:10,resizebi:6,resizeto:6,resolut:[1,6],resolv:0,respect:[11,14],rest:[11,13,14],result:11,retriev:[5,6,8,11,14,21],retriv:5,return_channel:6,root:5,row:20,run:[0,2,11,13,14,15,17,18,19],s3:6,s:[0,5,11,13,14,17],same:[11,13,14],sampl:[6,10,17,19],sample_img:[11,13,14],sample_z:14,sampler:5,save:6,scale:6,schedul:[11,13,14],scriptmodul:[15,17,18,19],search:1,search_modul:21,second:[10,11,13,14,20],see:[1,5,11,13,14],seed:1,select:[10,11,14],select_channel:[6,8],self:[5,11,13,14],sequenc:[5,6,7,11,14,17,18],sequenti:[3,4,11,13,14],serotini:[0,2],set:[0,1,5,6,10,11,13,14],setup:[2,5],sgd:[11,13,14],shape:[6,10],share:[15,17,18,19],should:[6,10,11,13,14,15,17,18,19],shown:[11,13,14],shuffl:5,shuffle_imag:6,sigma1:10,sigma2:10,signifi:6,silent:[15,17,18,19],similar:[11,13,14],simpli:4,sinc:[15,17,18,19],singl:[5,11,13,14],size:[5,6,10,11,13,14,17,18],skip:[11,13,14],smaller:5,smooth:[11,13,14],so:[10,11,13,14],softmax:11,some:[11,13,14,21],some_result:[11,13],someth:[11,13,14],sort:10,sourc:[4,5,6,7,8,9,10,11,13,14,15,17,18,19,20,21],space:17,spatial_pyramid_pool:[4,15],spatialpyramidpool:18,specif:[5,11,13,14],specifi:[1,11,13,14,17],split:[1,3,4,6,10],split_batches_for_dp:11,split_col:5,split_dataset:7,split_numb:20,splitdatamodul:5,sqrt:10,stabl:[1,10],startswith:[7,8],state:[5,11,13,14,15,17,18,19],statist:10,step:[1,11,13,14],still:11,store:6,store_metadata:11,store_model:11,str:[5,6,7,11,13,14,17,21],strict:[11,13,14],strictli:10,string:[11,14,17,18,21],structur:[1,13],sub_batch:11,subclass:[11,14,15,17,18,19],submit:0,submodul:[3,4],subpackag:3,subset:6,subset_channel:6,subset_train:5,sum:[11,13,14],support:[5,11,13,14],suppress:6,supress:6,sure:0,sutherland:10,swap:[3,4],swapax:20,symmetr:18,t:[2,5,11,13,14],t_co:[5,6,7],t_max:[11,13,14],tabular:14,tabular_conditional_va:[4,11],tabular_condprior_va:[4,11],tabular_va:[4,11],tabularconditionalpriorva:[11,14],tabularconditionalva:[11,14],tabularva:[11,14],tag:[0,11],take:[6,15,17,18,19],tarbal:2,target:17,target_dim:6,task:1,tell:[11,13,14,17],tensor:[5,6,10,11,13,14],tensorboard:11,term:[11,14],termin:2,test:[0,1,5,10,11,13,14],test_acc:[11,13,14],test_batch:[11,13,14],test_data:[11,13,14],test_dataload:5,test_epoch_end:[11,13,14],test_image_output:11,test_loss:[11,13,14],test_out:[11,13,14],test_step:[5,11,13,14],test_step_end:[11,13],test_step_out:[11,13],test_step_output:[11,13],text:[11,13,14],tf:11,than:6,thei:0,them:[6,15,17,18,19],thi:[0,1,2,4,5,6,7,11,13,14,15,17,18,19],thing:[1,11,13,14],those:[11,13,14],through:[0,2,11,13,14,17],tiff:[0,6],tiff_load:6,tiff_loader_czyx:6,tiff_writ:6,time:[11,13,14],to_tensor:20,todo:6,tolstikhin:10,torch:[5,6,7,9,10,11,13,14,15,17,18,19],torchvis:[11,13,14],totensor:5,tpu:[11,13,14],tr:10,track:10,train:[5,10,11,13,14,15,17,18,19],train_batch:11,train_data:11,train_dataload:5,train_out:11,trainer:[1,5,11],training_epoch_end:11,training_step:[11,13,14],training_step_end:11,training_step_output:11,transform:[3,4,5,8],transforms_config:8,trigger:1,truncat:[11,13,14],truncated_bptt_step:[11,13,14],tupl:[5,6,11,13,14],two:[1,10,11,13,14],type:[6,7,11,14],under:6,unet:[3,4],unetmodel:11,union:[5,6,7,11,14,18],unit:[11,13,14],unless:5,up:[0,11],up_residu:17,updat:[11,13,14],upresiduallay:17,us:[1,4,5,6,7,8,10,11,13,14,17],usag:15,use_amp:[11,13,14],util:[3,4,5,11,13,14],v:11,vae:[4,11,15],val:[5,11,13,14],val_acc:[11,13,14],val_batch:[11,13,14],val_data:[11,13,14],val_dataload:5,val_loss:[11,13,14],val_out:[11,13,14],val_step_output:11,valid:[1,5,11,13,14],validation_epoch_end:[11,13,14],validation_step:[5,11,13,14],validation_step_end:[11,13,14],valu:[5,6,8,10,11,13,14,20],variabl:[10,21],varianc:[11,14],variat:[11,13,14],vector:5,verbos:10,version:[0,4,10,17],version_str:11,via:5,virtualenv:0,visit:1,wae:10,want:11,warn:6,wasserstein:[11,13,14],we:7,websit:0,weight:[11,14],weight_init:[3,4],welcom:0,what:[11,13,14,17,18],whatev:[11,13,14],when:[0,1,5,6,10,11,13,14],where:5,whether:[5,6,11,13,14,17,18],which:[1,5,6,7,10,11,13,14],whose:[11,13,14],wi:10,width:10,wish:11,within:[15,17,18,19],without:[10,11,13,14],won:[11,13],work:[0,17],worker:5,workflow:1,wrap:7,x1:15,x2:15,x:[9,10,11,13,14,15,17,18,19],x_1:10,x_2:10,x_class:17,x_cond:13,x_dim:[5,11,14,17,19],x_label:[5,11,13,14],x_ref:17,x_target:17,y:[5,11,13,14],y_dim:5,y_encoded_label:8,y_label:[5,11],yaml:15,you:[0,2,5,11,13,14],your:[0,2,11,13,14],your_development_typ:0,your_name_her:0,yourself:5,z:[11,13,14,17,19],z_infer:13,z_target:17,zoo:[1,3,4],zyx:6,zyx_resolut:6},titles:["Contributing","Welcome to serotiny\u2019s documentation!","Installation","serotiny","serotiny package","serotiny.datamodules package","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.loaders package","serotiny.losses package","serotiny.metrics package","serotiny.models package","serotiny.models.callbacks package","serotiny.models.deprecated package","serotiny.models.vae package","serotiny.networks package","serotiny.networks.classification package","serotiny.networks.deprecated package","serotiny.networks.layers package","serotiny.networks.vae package","serotiny.transform package","serotiny.utils package"],titleterms:{"class":8,To:1,abstract_load:8,activ:18,base_va:14,buffered_patch_dataset:6,calculate_blur:10,calculate_fid:10,callback:12,cbvae:13,cbvae_decod:17,cbvae_decoder_mlp:[17,19],cbvae_encod:17,cbvae_encoder_conv:19,cbvae_encoder_mlp:[17,19],cbvae_mlp:13,classif:[11,16],column:8,conditional_va:14,condprior_va:14,config:1,content:[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],contribut:0,datafram:[7,8],dataframe_dataset:7,datamodul:[1,5],deploi:0,deprec:[13,17],develop:1,document:1,dummi:5,dynamic_import:21,elbo:9,featur:1,from:[1,2],gaussian:5,get:0,given:1,imag:6,image2d:8,image3d:8,image_va:14,incept:10,indic:1,instal:[1,2],io:[6,7,8],layer:18,load:1,loader:8,loss:9,manifest_datamodul:5,metric:10,mlp:15,model:[1,11,12,13,14],modul:[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],network:[15,16,17,18,19],packag:[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],pad:[18,20],patch:5,quick:1,regress:11,releas:2,s:1,sequenti:15,serotini:[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],setup:1,sourc:2,spatial_pyramid_pool:18,split:[5,7],stabl:2,start:[0,1],submodul:[5,6,7,8,9,10,11,13,14,15,17,18,19,20,21],subpackag:[4,6,7,11,15],swap:20,tabl:1,tabular_conditional_va:14,tabular_condprior_va:14,tabular_va:14,train:1,transform:[6,20],unet:11,util:[6,7,8,21],vae:[14,19],weight_init:15,welcom:1,zoo:11}})