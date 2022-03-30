Search.setIndex({docnames:["cli","dataframe_transforms","getting_started","index","installation","modules","serotiny","serotiny.datamodules","serotiny.datamodules.manifest_datamodule","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.dataframe_dataset","serotiny.io.dataframe.loaders","serotiny.io.dataframe.loaders.abstract_loader","serotiny.io.dataframe.loaders.classes","serotiny.io.dataframe.loaders.columns","serotiny.io.dataframe.loaders.image","serotiny.io.dataframe.loaders.random_image","serotiny.io.dataframe.readers","serotiny.io.image","serotiny.losses","serotiny.losses.continuous_bernoulli","serotiny.losses.kl_divergence","serotiny.ml_ops","serotiny.ml_ops.ml_ops","serotiny.ml_ops.mlflow_utils","serotiny.ml_ops.utils","serotiny.models","serotiny.models.base_model","serotiny.models.basic_model","serotiny.models.utils","serotiny.models.utils.optimizer_utils","serotiny.models.vae","serotiny.models.vae.base_vae","serotiny.models.vae.image_vae","serotiny.models.vae.tabular_vae","serotiny.networks","serotiny.networks.basic_cnn","serotiny.networks.basic_cnn.basic_cnn","serotiny.networks.layers","serotiny.networks.layers.convolution_block","serotiny.networks.layers.skip_connection","serotiny.networks.layers.spatial_pyramid_pool","serotiny.networks.mlp","serotiny.networks.mlp.mlp","serotiny.networks.utils","serotiny.networks.utils.weight_init","serotiny.transforms","serotiny.transforms.dataframe","serotiny.transforms.dataframe.transforms","serotiny.transforms.image","serotiny.transforms.image.crop","serotiny.transforms.image.normalize","serotiny.transforms.image.pad","serotiny.transforms.image.project","serotiny.transforms.image.swap"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","dataframe_transforms.rst","getting_started.rst","index.rst","installation.rst","modules.rst","serotiny.rst","serotiny.datamodules.rst","serotiny.datamodules.manifest_datamodule.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.dataframe_dataset.rst","serotiny.io.dataframe.loaders.rst","serotiny.io.dataframe.loaders.abstract_loader.rst","serotiny.io.dataframe.loaders.classes.rst","serotiny.io.dataframe.loaders.columns.rst","serotiny.io.dataframe.loaders.image.rst","serotiny.io.dataframe.loaders.random_image.rst","serotiny.io.dataframe.readers.rst","serotiny.io.image.rst","serotiny.losses.rst","serotiny.losses.continuous_bernoulli.rst","serotiny.losses.kl_divergence.rst","serotiny.ml_ops.rst","serotiny.ml_ops.ml_ops.rst","serotiny.ml_ops.mlflow_utils.rst","serotiny.ml_ops.utils.rst","serotiny.models.rst","serotiny.models.base_model.rst","serotiny.models.basic_model.rst","serotiny.models.utils.rst","serotiny.models.utils.optimizer_utils.rst","serotiny.models.vae.rst","serotiny.models.vae.base_vae.rst","serotiny.models.vae.image_vae.rst","serotiny.models.vae.tabular_vae.rst","serotiny.networks.rst","serotiny.networks.basic_cnn.rst","serotiny.networks.basic_cnn.basic_cnn.rst","serotiny.networks.layers.rst","serotiny.networks.layers.convolution_block.rst","serotiny.networks.layers.skip_connection.rst","serotiny.networks.layers.spatial_pyramid_pool.rst","serotiny.networks.mlp.rst","serotiny.networks.mlp.mlp.rst","serotiny.networks.utils.rst","serotiny.networks.utils.weight_init.rst","serotiny.transforms.rst","serotiny.transforms.dataframe.rst","serotiny.transforms.dataframe.transforms.rst","serotiny.transforms.image.rst","serotiny.transforms.image.crop.rst","serotiny.transforms.image.normalize.rst","serotiny.transforms.image.pad.rst","serotiny.transforms.image.project.rst","serotiny.transforms.image.swap.rst"],objects:{"":[[6,0,0,"-","serotiny"]],"serotiny.datamodules":[[7,1,1,"","ManifestDatamodule"],[8,0,0,"-","manifest_datamodule"]],"serotiny.datamodules.ManifestDatamodule":[[7,2,1,"","make_dataloader"],[7,2,1,"","predict_dataloader"],[7,2,1,"","test_dataloader"],[7,2,1,"","train_dataloader"],[7,2,1,"","val_dataloader"]],"serotiny.datamodules.manifest_datamodule":[[8,1,1,"","ManifestDatamodule"]],"serotiny.datamodules.manifest_datamodule.ManifestDatamodule":[[8,2,1,"","make_dataloader"],[8,2,1,"","predict_dataloader"],[8,2,1,"","test_dataloader"],[8,2,1,"","train_dataloader"],[8,2,1,"","val_dataloader"]],"serotiny.io":[[10,0,0,"-","dataframe"],[19,0,0,"-","image"]],"serotiny.io.dataframe":[[11,0,0,"-","dataframe_dataset"],[12,0,0,"-","loaders"],[18,0,0,"-","readers"]],"serotiny.io.dataframe.dataframe_dataset":[[11,1,1,"","DataframeDataset"]],"serotiny.io.dataframe.loaders":[[12,1,1,"","LoadClass"],[12,1,1,"","LoadColumn"],[12,1,1,"","LoadColumns"],[12,1,1,"","LoadImage"],[13,0,0,"-","abstract_loader"],[14,0,0,"-","classes"],[15,0,0,"-","columns"],[16,0,0,"-","image"],[17,0,0,"-","random_image"]],"serotiny.io.dataframe.loaders.abstract_loader":[[13,1,1,"","Loader"]],"serotiny.io.dataframe.loaders.classes":[[14,1,1,"","LoadClass"]],"serotiny.io.dataframe.loaders.columns":[[15,1,1,"","LoadColumn"],[15,1,1,"","LoadColumns"]],"serotiny.io.dataframe.loaders.image":[[16,1,1,"","LoadImage"]],"serotiny.io.dataframe.loaders.random_image":[[17,1,1,"","LoadRandomTensor"]],"serotiny.io.dataframe.readers":[[18,3,1,"","filter_columns"],[18,3,1,"","read_csv"],[18,3,1,"","read_dataframe"],[18,3,1,"","read_parquet"]],"serotiny.io.image":[[19,3,1,"","image_loader"],[19,3,1,"","infer_dims"],[19,3,1,"","tiff_writer"]],"serotiny.losses":[[21,0,0,"-","continuous_bernoulli"],[22,0,0,"-","kl_divergence"]],"serotiny.losses.continuous_bernoulli":[[21,1,1,"","CBLogLoss"]],"serotiny.losses.continuous_bernoulli.CBLogLoss":[[21,2,1,"","forward"],[21,4,1,"","reduction"]],"serotiny.losses.kl_divergence":[[22,3,1,"","diagonal_gaussian_kl"],[22,3,1,"","isotropic_gaussian_kl"]],"serotiny.ml_ops":[[24,0,0,"-","ml_ops"],[25,0,0,"-","mlflow_utils"],[26,0,0,"-","utils"]],"serotiny.ml_ops.mlflow_utils":[[25,3,1,"","load_model_from_checkpoint"],[25,3,1,"","mlflow_fit"],[25,3,1,"","mlflow_predict"],[25,3,1,"","mlflow_test"],[25,3,1,"","patched_autolog"]],"serotiny.ml_ops.utils":[[26,3,1,"","flatten_config"],[26,3,1,"","get_serotiny_project"],[26,3,1,"","make_notebook"],[26,3,1,"","save_model_predictions"]],"serotiny.models":[[28,0,0,"-","base_model"],[29,0,0,"-","basic_model"],[30,0,0,"-","utils"],[32,0,0,"-","vae"]],"serotiny.models.base_model":[[28,1,1,"","BaseModel"]],"serotiny.models.base_model.BaseModel":[[28,2,1,"","configure_optimizers"],[28,2,1,"","forward"],[28,2,1,"","parse_batch"],[28,2,1,"","predict_step"],[28,2,1,"","test_epoch_end"],[28,2,1,"","test_step"],[28,2,1,"","train_epoch_end"],[28,4,1,"","training"],[28,2,1,"","training_step"],[28,2,1,"","validation_epoch_end"],[28,2,1,"","validation_step"]],"serotiny.models.basic_model":[[29,1,1,"","BasicModel"]],"serotiny.models.basic_model.BasicModel":[[29,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[29,2,1,"","forward"],[29,2,1,"","parse_batch"],[29,4,1,"","precision"],[29,4,1,"","prepare_data_per_node"],[29,4,1,"","trainer"],[29,4,1,"","training"]],"serotiny.models.utils":[[31,0,0,"-","optimizer_utils"]],"serotiny.models.utils.optimizer_utils":[[31,3,1,"","find_lr_scheduler"],[31,3,1,"","find_optimizer"]],"serotiny.models.vae":[[33,0,0,"-","base_vae"],[34,0,0,"-","image_vae"],[35,0,0,"-","tabular_vae"]],"serotiny.models.vae.base_vae":[[33,1,1,"","BaseVAE"]],"serotiny.models.vae.base_vae.BaseVAE":[[33,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[33,2,1,"","calculate_elbo"],[33,2,1,"","forward"],[33,2,1,"","parse_batch"],[33,4,1,"","precision"],[33,4,1,"","prepare_data_per_node"],[33,2,1,"","sample_z"],[33,4,1,"","trainer"],[33,4,1,"","training"]],"serotiny.models.vae.image_vae":[[34,1,1,"","ImageVAE"]],"serotiny.models.vae.image_vae.ImageVAE":[[34,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[34,4,1,"","precision"],[34,4,1,"","prepare_data_per_node"],[34,4,1,"","trainer"],[34,4,1,"","training"]],"serotiny.models.vae.tabular_vae":[[35,1,1,"","TabularVAE"]],"serotiny.models.vae.tabular_vae.TabularVAE":[[35,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[35,4,1,"","precision"],[35,4,1,"","prepare_data_per_node"],[35,4,1,"","trainer"],[35,4,1,"","training"]],"serotiny.networks":[[37,0,0,"-","basic_cnn"],[39,0,0,"-","layers"],[43,0,0,"-","mlp"],[45,0,0,"-","utils"]],"serotiny.networks.basic_cnn":[[38,0,0,"-","basic_cnn"]],"serotiny.networks.basic_cnn.basic_cnn":[[38,1,1,"","BasicCNN"]],"serotiny.networks.basic_cnn.basic_cnn.BasicCNN":[[38,2,1,"","conv_forward"],[38,2,1,"","forward"],[38,4,1,"","training"]],"serotiny.networks.layers":[[40,0,0,"-","convolution_block"],[41,0,0,"-","skip_connection"],[42,0,0,"-","spatial_pyramid_pool"]],"serotiny.networks.layers.convolution_block":[[40,1,1,"","ConvBlock"],[40,3,1,"","conv_block"]],"serotiny.networks.layers.convolution_block.ConvBlock":[[40,2,1,"","forward"],[40,4,1,"","training"]],"serotiny.networks.layers.skip_connection":[[41,1,1,"","SkipConnection"]],"serotiny.networks.layers.skip_connection.SkipConnection":[[41,2,1,"","forward"],[41,4,1,"","training"]],"serotiny.networks.layers.spatial_pyramid_pool":[[42,1,1,"","SpatialPyramidPool"],[42,3,1,"","spatial_pyramid_pool"]],"serotiny.networks.layers.spatial_pyramid_pool.SpatialPyramidPool":[[42,2,1,"","forward"],[42,4,1,"","training"]],"serotiny.networks.mlp":[[44,0,0,"-","mlp"]],"serotiny.networks.mlp.mlp":[[44,1,1,"","MLP"]],"serotiny.networks.mlp.mlp.MLP":[[44,2,1,"","forward"],[44,4,1,"","training"]],"serotiny.networks.utils":[[46,0,0,"-","weight_init"]],"serotiny.networks.utils.weight_init":[[46,3,1,"","weight_init"]],"serotiny.transforms":[[48,0,0,"-","dataframe"],[50,0,0,"-","image"]],"serotiny.transforms.dataframe":[[49,0,0,"-","transforms"]],"serotiny.transforms.dataframe.transforms":[[49,3,1,"","append_class_weights"],[49,3,1,"","append_labels_to_integers"],[49,3,1,"","append_one_hot"],[49,3,1,"","filter_columns"],[49,3,1,"","filter_rows"],[49,3,1,"","make_random_df"],[49,3,1,"","sample_n_each"],[49,3,1,"","split_dataframe"]],"serotiny.transforms.image":[[51,0,0,"-","crop"],[52,0,0,"-","normalize"],[53,0,0,"-","pad"],[54,0,0,"-","project"],[55,0,0,"-","swap"]],"serotiny.transforms.image.crop":[[51,1,1,"","CropCenter"]],"serotiny.transforms.image.normalize":[[52,1,1,"","NormalizeAbsolute"],[52,1,1,"","NormalizeMean"],[52,1,1,"","NormalizeMinMax"]],"serotiny.transforms.image.pad":[[53,1,1,"","ExpandColumns"],[53,1,1,"","ExpandTo"],[53,1,1,"","PadTo"],[53,3,1,"","expand_columns"],[53,3,1,"","expand_to"],[53,3,1,"","pull_to"],[53,3,1,"","split_number"],[53,3,1,"","to_tensor"]],"serotiny.transforms.image.project":[[54,1,1,"","Project"]],"serotiny.transforms.image.swap":[[55,1,1,"","Permute"],[55,1,1,"","SwapAxes"]],serotiny:[[7,0,0,"-","datamodules"],[9,0,0,"-","io"],[20,0,0,"-","losses"],[23,0,0,"-","ml_ops"],[27,0,0,"-","models"],[36,0,0,"-","networks"],[47,0,0,"-","transforms"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[1,2,22,25,34,35,40,51,52,53],"06845":21,"1":[1,2,22,25,34,35,49,52],"10":[2,25],"100":[2,49],"11":0,"1234":2,"13":0,"1337":2,"16":2,"1907":21,"1e":2,"2":[1,2,52],"20":2,"256":44,"2d":40,"3":[1,2,38,40],"30":2,"32":2,"3d":[34,38,40],"4":1,"42":49,"5":[2,25],"50":[0,1],"64":2,"7":1,"abstract":13,"boolean":33,"case":[2,7,8,49],"class":[2,7,8,10,11,12,13,15,16,17,21,28,29,33,34,35,38,40,41,42,44,49,51,52,53,54,55],"default":[0,2,40,49],"do":[0,2,3],"final":2,"float":[2,12,15,33,34,35,49],"function":[0,2,11,29,33,40,49],"import":2,"int":[2,12,14,17,19,29,33,34,35,38,40,49,53,55],"long":2,"new":2,"null":2,"public":4,"return":[2,12,14,18,19,49],"true":[2,7,8,12,16,19,34,38,40,49,51],"try":2,A:[2,7,8,11,12,15,18,29,49],AND:18,And:38,At:2,But:[12,16],By:49,For:[0,1,2,12,16,18],If:[2,4,7,8,12,15,18,19,49],In:[0,2,3,11,49],It:[1,2,3,7,8,12,15,17,22,49],Of:2,On:2,One:2,Or:4,That:[0,1,2],The:[0,4,7,8,11,12,15,19,29,49],There:2,These:[0,2],To:[1,2,4],With:2,_:[2,3],_loss:[21,29,33,34],_partial_:2,_step:2,_target_:2,a_datamodule_config:0,a_model_config:0,a_tiny_project:2,ab:21,abl:2,about:[0,2],abov:[0,2],abstract_load:[7,8,10,12,14,15,16,17],achiev:3,ad:2,adam:[2,29,33,34,35],add:[0,2,49],addit:[0,7,8],addition:[2,7,8],adopt:3,afford:3,ahead:2,aic:[2,3,11],aics_imag:19,aicsimag:19,aicsimageio:[12,16,19],all:[1,2,18,49],allencellmodel:[2,4],allow:1,allow_zero_length_dataloader_with_multiple_devic:[29,33,34,35],also:[0,1,18,22],alwai:[2,4,13],among:[7,8],an:[0,2,12,15,19,22],ani:[0,2],anisotrop:33,append:[0,49],append_class_weight:49,append_labels_to_integ:49,append_one_hot:49,appli:[2,12,16,18,19,38,40,46],applic:2,appropri:[2,11],ar:[0,2,7,8,12,13,15,18,49],aren:[0,18],arg:[7,8],argument:[0,1,2,7,8,18,49],around:[2,29],arrai:[12,15,16,19,33,34,35,53],artifact:[2,3],arxiv:21,asid:[7,8],assert:18,assign:19,assum:2,astyp:[12,16],autolog:25,avail:2,ax:53,axi:[52,54],b:[1,2,49],backend:2,balanc:49,base:[2,7,8,11,12,13,14,15,16,17,21,28,29,33,34,35,38,40,41,42,44,49,51,52,53,54,55],base_model:[6,27,29,33],base_va:[27,32,34,35],basemodel:[28,29,33],baseva:[33,34,35],bash:0,basic:33,basic_cnn:[6,36],basic_model:[6,27],basiccnn:38,basicmodel:[2,29],batch:[2,22,28,29,33],batch_idx:28,batch_norm:[34,38,40],batch_siz:[2,7,8],becaus:0,becom:[2,3],befor:19,behavior:[0,2],behind:0,belong:49,benefit:3,bernoulli:21,best:3,beta:[33,34,35],better:2,between:[22,49],big:2,binari:[12,14],bit:2,block:[2,40],boil:2,bolt:2,bool:[7,8,12,14,16,19,28,29,33,34,35,38,40,41,42,44,49],build:2,c:[1,49],cache_output:[33,34,35],calcul:2,calculate_elbo:33,call:[0,2,25,49],callabl:[12,13,16,19,29],callback:25,calll:2,can:[0,1,2,3,4,7,8,33,40,49],cblogloss:21,cell:[0,2,11],cell_id:2,cell_typ:2,cellid:1,center:2,center_of_mass:51,cfg:26,chang:[0,2],channel:[12,16,19,38,40],channel_nam:19,check:[0,2],classifi:2,cli:[1,2],clip_max:52,clip_min:52,clip_quantil:52,clone:4,cnn:38,collabor:3,collect:2,column:[1,2,7,8,10,11,12,14,16,17,18,49,53],columns_to_filt:18,com:[2,4],combin:18,come:0,command:[0,2,4],common:1,commun:2,compat:[2,25],complet:0,compress:2,comput:[2,22],compute_loss:33,concaten:[12,15],condit:18,config:0,configur:[0,3],configure_optim:28,consist:[11,17],constant:53,construct:[2,7,8],consum:2,contain:[2,7,8,11,12,14,15,16,18,19,49],content:11,context:0,continu:21,continuous_bernoulli:[6,20],contribut:2,conv_block:40,conv_forward:38,convblock:40,convert:49,convolut:40,convolution_block:[36,39],cookiecutt:2,copi:4,core:[7,8,18,28,49],correct:19,correspond:[2,11,19,49],could:2,coupl:2,cours:2,covari:33,cover:25,creat:[0,1,2,7,8,49],crop:[47,50],crop_raw:1,crop_seg:1,cropcent:51,cropi:51,cropx:51,cropz:51,cross:2,crossentropyloss:2,csv:[1,2,18],curl:4,current:33,custom:2,d:[2,49],dashboard:2,data:[0,7,8,11,13,25,40,49],datafram:[0,2,6,7,8,9,47],dataframe_dataset:[9,10],dataframedataset:[11,13],dataload:[2,7,8,29],dataloader_kwarg:[7,8],datamodul:[2,5,6],dataset:[2,7,8,11,49],decod:33,decoder_non_linear:34,deep:3,defin:1,depth:38,desir:18,destin:[0,2],detail:49,determin:[12,14,19,33],develop:[2,3],diagon:[22,33],diagonal_gaussian_kl:22,dict:[7,8,11],differ:[0,2,18,49],dim:[17,44],dim_ord:19,dimens:[17,19,38,40,53],dimension:[38,40],directli:[2,12,15],directori:2,disabl:25,disable_for_unsupported_vers:25,disk:19,diverg:22,dl:3,dna:2,doc:[0,2,7,8,12,16],doe:2,doesn:[2,19],don:[2,4],done:[0,1],down:2,download:4,drive:[2,3],dtype:[2,12,15,16,19],dummi:17,dump:2,dure:2,e:[0,2,18,33],each:[0,2,7,8,11,19,38,49],earli:2,earlystop:2,easi:3,easier:2,either:[2,4,7,8,18,33,49],element:2,els:2,enabl:[12,15,25,49],enable_checkpoint:2,encod:[33,49],end:[1,2,12,15,18,49],endswith:[12,15,18,49],enough:2,entropi:2,equal:49,etc:[2,7,8],eval:[0,2],evalu:[0,2],even:2,everyth:[7,8],exampl:[0,2],exclud:[1,12,15,18,49],exclus:25,execut:3,exist:49,expand_column:53,expand_to:53,expandcolumn:53,expanded_column:53,expandto:53,expect:[2,18],experi:[1,2,3],experiment_nam:2,explicit:[12,15,49],expos:1,express:[12,15,18,49],extend:49,extract:13,fact:2,fail:2,fals:[1,7,8,12,14,19,25,33,34,35,38,40,49,52],familiar:2,fashion:2,featur:2,feed:2,fetch:2,few:2,field:[12,15],file1:2,file2:2,file3:2,file:[0,1,2,7,8,11,12,16,18],file_typ:[12,16],filter:[1,12,15,18,49],filter_column:[1,12,15,18,49],filter_row:[1,49],final_non_linear:38,find_lr_schedul:31,find_optim:31,first:[1,2,22,55],fit:25,fix:33,flag:[0,2,7,8,12,14,19,33],flat_output:38,flatten_config:26,float32:[12,16,19],fold:[7,8],folder:2,follow:[2,25],forc:49,force_s:51,form:2,format:[12,16],forward:[2,21,28,29,33,38,40,41,42,44],found:18,fraction:49,frame:[18,49],framework:[2,3],from:[0,2,3,7,8,11,12,13,14,15,16,18,19,29,31,33],full_conf:25,futur:[12,16],g:[0,2],gaussian:22,gener:[1,2,3,12,16,17,29,49],get:[0,31],get_serotiny_project:26,git:4,github:[2,4],given:[0,2,7,8,12,16,18,19,31,49],go:[2,7,8],goal:3,govern:2,group:0,guid:4,ha:[0,2,49],half:49,have:[0,2,4,19,49],heavili:2,help:0,helper:1,here:[0,1,2,21],hidden:38,hidden_channel:[34,38],hidden_lay:[2,35,44],highli:3,hood:0,hot:[12,14,49],how:[0,2],html:[7,8,12,16],http:[2,3,4,7,8,12,16,21],hydra:[0,2,3],hyperparamet:2,i:[2,18,33],id:2,id_label:[33,34,35],identifi:[2,49],ignor:[2,7,8,12,15,17,18,49],imag:[0,2,6,9,10,11,12,47],image_load:19,image_path:2,image_va:[27,32],imageva:34,img:19,implement:[1,2,13],implicitli:38,importantli:1,in_c:40,in_channel:[34,38],includ:[12,15,16,18,49],include_column:18,incomplet:2,independ:2,index:19,infer:[0,2,7,8,19],infer_dim:19,info:[0,2,7,8,12,16],inform:[0,2],initi:[0,33],inner:2,input:[1,2,21,29,38,40,49],input_dim:[34,38],input_s:2,instal:[0,2],instanc:[2,13],instanti:[2,7,8,33,38,40],instead:2,integ:[19,49],integr:[2,3],intend:0,intent:3,interact:[0,2],interfac:[2,33],interpol:2,intimid:0,involv:1,io:[2,5,6,7,8],ipynb:[0,2],isn:2,isotrop:[22,33,34,35],isotropic_gaussian_kl:22,its:[0,2,13,49],joblib:2,jupyt:2,just:[0,2,7,8],just_infer:[2,7,8],keep:49,kei:[2,11,29],kept:17,kernel:40,kernel_s:[38,40],keyword:[7,8],kl_diverg:[6,20],kld:33,know:1,known:25,kullback:22,kwarg:[28,29,33],label:[2,19,33],latenc:18,latent_dim:[0,2,33,34,35],later:2,launcher:0,layer:[6,36,38],learn:[0,2,3,33],learn_prior_logvar:[33,34,35],leibler:22,less:49,let:1,level:2,leverag:[0,2,12,15],lifecycl:3,lightiningdatamodul:2,lightn:[2,3,7,8,25,28,29],lightningdatamodul:[7,8],lightningmodul:[2,28],like:[0,2,12,16],line:[0,2],list:[2,7,8,12,15,16,18,19,49],ll:0,load:[7,8,11,12,13,16,17,18,19],load_as_torch:[12,16],load_model_from_checkpoint:[2,25],loadclass:[12,14],loadcolumn:[2,12,15],loader:[2,7,8,9,10,11],loadimag:[2,12,16],loadrandomtensor:17,log:[22,33],log_every_n_epoch:25,log_model:25,log_var:22,logic:[2,18],logvar1:22,logvar2:22,logvar:[22,33],look:2,loop:2,loss:[2,5,6,29,33,34],loss_mask_label:[33,34,35],lr:2,lr_schedul:31,m:[0,46],machin:2,made:[7,8],mai:[2,25],main:[2,4],make:[0,3],make_dataload:[7,8],make_notebook:[0,2,26],make_random_df:[1,49],manag:3,mani:0,manifest:[1,2,7,8,11,13],manifest_datamodul:[6,7],manifestdatamodul:[7,8],map:19,mask:33,match:[12,15,18,49],matrix:33,max:[38,54],max_epoch:[0,2],max_pool_lay:[34,38],mean:[0,2,22,34],mechan:13,medium:2,merg:1,method:[2,4,11],metric:2,might:[2,12,16],mimic:[0,2],min_delta:2,minim:29,ml:2,ml_op:[2,5,6],mlflow:[0,3,25],mlflow_conf:25,mlflow_fit:25,mlflow_predict:25,mlflow_test:25,mlflow_util:[2,6,23],mlp:[6,36],mode:[21,25,34,38,40,53,54],model:[0,5,6,25,26,46],modifi:49,modul:[0,1,2,6,7,9,10,12,20,23,27,30,32,36,37,39,43,45,47,48,50],modular:[2,3],modularli:2,monitor:2,more:[0,2,7,8,12,16,49],more_param:2,most:4,mostli:1,mseloss:[33,34],mu1:22,mu2:22,mu:[22,33],multi:0,multipl:[0,12,15],must:[12,15,18,19,49],my:2,my_input:1,my_output:1,mycustommodel:2,mycustomnetwork:2,n:53,n_row:[1,49],name:[2,7,8,12,14,18,19,31],ndarrai:[12,16],necessarili:22,need:[1,2],net:3,network:[2,5,6,29,33],neural:3,newli:2,nn:[2,21,29,33,34,38,40,41,42,44],non_linear:[34,38,40],none:[7,8,12,15,16,18,19,29,33,34,35,38,40,44,49,51,52,53],noqa:[12,16],normal:[2,11,47,50],normalizeabsolut:52,normalizemean:52,normalizeminmax:52,note:[0,1,2],notebook:[0,2],notion:2,now:[0,2,12,16],np:[12,16,19],num_class:[12,14],num_work:[2,7,8],number:[12,14,16,19,38,40,49],numpi:[12,16,19,33,34,35],object:[2,13,51,52,53,54,55],obtain:0,ol:4,omegaconf:2,omit:[0,2],on_test_epoch_end:25,onc:[0,4],one:[0,2,7,8,12,14,49],one_hot_encod:49,oni:2,onli:[0,2,7,8,12,16,18,19,33,49],oper:[2,38,49],optim:[2,29,31,33,34,35],optimizer_nam:31,optimizer_util:[27,30],option:[0,7,8,12,15,16,18,19,29,33,34,35,38,40,49],order:[19,52],org:[3,7,8,12,16,21],organ:2,other:[2,12,16,17],otherwis:18,our:2,out:2,out_c:40,out_pool_s:42,output:[19,28,38,40],output_dim:38,output_dir:[2,26],output_dtyp:19,output_path:1,outsid:[2,25],over:2,overrid:[0,2],p:2,packag:[2,5,25],package_nam:2,pad:[40,47,50,51],padto:53,panda:[11,13,18,49],param1:2,param2:2,param3:2,paramet:[0,2,7,8,11,12,14,15,16,17,18,19,22,29,33,38,40,49],parquet:[7,8,18],pars:2,parse_batch:[2,28,29,33],part:2,particular:0,partit:1,pass:[7,8],patch:25,patched_autolog:25,path:[0,2,7,8,11,12,16,18,19,26],pathlib:[7,8,18,19],patienc:2,pattern:18,pd:[11,18,49],per:49,perform:49,permut:55,pip:[2,4],pipelin:49,pl:[29,33,34,35],place:2,placehold:1,png:2,point:[1,11],pool:38,popul:[0,2],port:2,possibl:[2,12,14],practic:[3,11],preced:[1,18],precis:[29,33,34,35],pred:[2,26],predict:[3,7,8,29],predict_dataload:[2,7,8],predict_datamodul:[7,8],predict_step:[2,28],prefer:4,prepare_data_per_nod:[29,33,34,35],previou:[1,2],prior:33,prior_logvar:[33,34,35],prior_mod:[33,34,35],prob:21,process:[2,4],produc:2,programat:2,project:[0,3,47,50],project_descript:2,project_nam:2,propos:21,provid:[1,13,49],pull_to:53,purpos:2,py:[2,4],pyramid_pool_split:38,python:[2,4],pytorch:[2,3,7,8,11,25,29,33],pytorch_lightn:[2,7,8,28],queri:[12,15,18],random:[1,49],random_imag:[10,12],randomli:17,rang:25,rather:[12,16],re:[0,1],read:[2,18],read_csv:18,read_datafram:18,read_parquet:18,reader:[9,10,12,16,19],readi:2,reason:[0,2],recent:4,recogn:2,recommend:2,reconstruct:33,reconstruction_loss:[33,34],reconstruction_reduc:[33,34],reduct:21,refer:[2,12,16],regardless:[7,8],regex:[12,15,18,49],regular:[12,15,18,49],relat:2,relev:0,reli:[2,3],remain:[2,12,15,18,49],rememb:2,replac:49,repo:4,repositori:4,reproduc:3,requir:[2,12,16,18],required_column:18,respect:[0,2,18,33],restart:2,result:[1,2,12,15,18,19,29,49],retriev:[12,14,15,16,19,29,33],return_as_torch:19,return_channel:19,return_s:38,return_split:[1,49],return_torch:52,right:2,row:[1,2,13,49,53],run:[0,2,4],run_id:25,run_nam:2,s:[0,2,7,8,49],same:[2,33],sampl:49,sample_n_each:49,sample_z:33,save:[2,19,29],save_model_predict:26,save_predict:[2,29],sc:0,scale:52,scenario:2,schedul:31,scheduler_nam:31,script:0,second:[1,22,55],section:[0,2],see:[0,2,7,8,12,16,49],seed:[0,2,49],select:[2,12,15,49],select_channel:[2,12,16,19],separ:49,sequenc:[7,8,12,15,16,17,18,19,33,34,35,38,40,49],serotini:[1,2,4],serotiny_bash:0,server:2,session:0,set:[2,3,7,8],setup:[2,4],shall:13,shell:0,should:2,shuffl:[7,8],signifi:1,silent:25,simpl:[2,12,15],simpli:[2,7,8],simplist:2,singl:[2,7,8],skip_connect:[34,36,38,39,40],skipconnect:41,slower:18,small:2,smaller:49,so:[0,2,3,7,8],some:[2,12,16],some_run:2,someon:2,someth:2,sourc:[7,8,11,12,13,14,15,16,17,18,19,21,22,25,26,28,29,31,33,34,35,38,40,41,42,44,46,49,51,52,53,54,55],spatial_pyramid_pool:[36,39],spatialpyramidpool:42,specif:[2,7,8,13],specifi:[0,2],split:[1,2,7,8,49],split_column:[2,7,8],split_datafram:[1,49],split_numb:53,stabl:[7,8,12,16],stand:2,standard:[2,3],start:[0,1,12,15,18,49],startswith:[12,15,18,49],step:[1,2],stop:2,store:[2,18],str:[7,8,12,14,15,16,17,18,19,21,29,33,34,35,38,40,49],straightforward:0,stratifi:49,streamlin:3,string:[2,12,15,18,33,49],structur:3,subclass:[2,13,33],submodul:[2,5,6,36,47],subpackag:5,substr:[12,15,18,49],succe:25,sum:[21,33],suppli:[2,12,15,18,49],support:[12,16,18,33],swap:[2,47,50],swapax:55,sweep:[0,2],syntax:[0,2],t:[0,2,4,18,19],tab:0,tabular_va:[27,32],tabularva:[2,35],take:[2,18],tarbal:4,target:[2,21,29],target_dim:[53,55],task:2,tell:2,templat:2,temporari:2,tensor:[12,16,17,19,22,33,34,35],term:33,termin:4,test:[2,3,7,8,25,33,34,35,49],test_dataload:[7,8],test_epoch_end:28,test_step:28,than:[12,16,49],the_run_id:2,the_tracking_uri:2,thei:2,them:[0,1,2,12,13,15],thi:[0,1,2,3,4,7,8,11,12,13,16,18,19,25,49],third:1,three:[2,7,8,49],through:[2,4],throughout:2,thrown:18,thu:2,tiff:[12,16,19],tiff_writ:19,tightli:3,tini:2,tinker:[0,2],to_tensor:53,todo:49,toggl:49,too:[2,12,16],tool:[1,3],top:2,torch:[2,7,8,11,12,16,19,21,22,29,31,33,34,35,38,40,41,42,44],track:3,tracking_uri:[2,25],train:[1,3,7,8,28,29,33,34,35,38,40,41,42,44,49],train_dataload:[7,8],train_epoch_end:28,train_frac:[1,49],trainer:[0,25,29,33,34,35],training_step:28,transform:[5,6,12,16,19],trivial:2,two:[2,22],type:[12,16,18,19,29,33,34,35],under:[0,2],underli:17,understand:[0,2],union:[7,8,12,16,18,19,33,34,35,38,49],uniqu:[2,49],unlik:2,up:2,up_conv:[38,40],upon:[12,16],upsampl:49,upsample_lay:38,uri:2,us:[0,1,3,7,8,11,12,13,14,15,16,17,18,19,25,29,33,49],usag:[2,46],user:1,usual:2,util:[0,6,7,8,11,23,27,36,40],vae:[6,27],val:[1,7,8,49],val_dataload:[7,8],val_frac:49,val_loss:2,valid:[2,49],validation_epoch_end:28,validation_step:28,valu:[0,2,7,8,12,14,33,40,49,53],valueerror:18,varianc:[22,33],ve:0,vector:[12,14],version:25,via:2,volum:2,wa:3,wai:2,want:[0,1,2],we:[1,2,11],weight:[33,49],weight_init:[36,45],well:[0,2,25],what:[0,2],when:[0,2,7,8,19,25],where:[2,19,49],wherea:2,whether:[7,8,12,14,16,19,33,49],which:[0,1,2,7,8,11,12,13,16,33,38,49],whole:[2,7,8],whose:2,wise:22,wish:2,within:2,without:2,work:[2,22],would:2,wrangl:[0,49],wrap:[11,29],wrapped_modul:41,wrapper:29,write:[2,19],x1:44,x2:44,x:[2,28,29,33,38,40,41,42],x_dim:[2,35],x_hat:33,x_label:[2,29,33,34,35],xz:2,y:[2,29],y_encoded_label:[12,14],y_label:[2,29],yaml:2,yhat:2,you:[0,1,2,4],your:[0,4],your_data_config_nam:2,your_model_config_nam:2,yourcustomcallback:2,zarr:[12,16]},titles:["serotiny CLI","Dataframe wrangling","Getting started","serotiny","Installation","serotiny","serotiny package","serotiny.datamodules package","serotiny.datamodules.manifest_datamodule module","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.dataframe_dataset module","serotiny.io.dataframe.loaders package","serotiny.io.dataframe.loaders.abstract_loader module","serotiny.io.dataframe.loaders.classes module","serotiny.io.dataframe.loaders.columns module","serotiny.io.dataframe.loaders.image module","serotiny.io.dataframe.loaders.random_image module","serotiny.io.dataframe.readers module","serotiny.io.image module","serotiny.losses package","serotiny.losses.continuous_bernoulli module","serotiny.losses.kl_divergence module","serotiny.ml_ops package","serotiny.ml_ops.ml_ops module","serotiny.ml_ops.mlflow_utils module","serotiny.ml_ops.utils module","serotiny.models package","serotiny.models.base_model module","serotiny.models.basic_model module","serotiny.models.utils package","serotiny.models.utils.optimizer_utils module","serotiny.models.vae package","serotiny.models.vae.base_vae module","serotiny.models.vae.image_vae module","serotiny.models.vae.tabular_vae module","serotiny.networks package","serotiny.networks.basic_cnn package","serotiny.networks.basic_cnn.basic_cnn module","serotiny.networks.layers package","serotiny.networks.layers.convolution_block module","serotiny.networks.layers.skip_connection module","serotiny.networks.layers.spatial_pyramid_pool module","serotiny.networks.mlp package","serotiny.networks.mlp.mlp module","serotiny.networks.utils package","serotiny.networks.utils.weight_init module","serotiny.transforms package","serotiny.transforms.dataframe package","serotiny.transforms.dataframe.transforms module","serotiny.transforms.image package","serotiny.transforms.image.crop module","serotiny.transforms.image.normalize module","serotiny.transforms.image.pad module","serotiny.transforms.image.project module","serotiny.transforms.image.swap module"],titleterms:{"class":14,The:2,abstract_load:13,architectur:2,base_model:28,base_va:33,basic_cnn:[37,38],basic_model:29,bring:2,call:1,callback:2,chain:1,cli:0,column:15,config:2,configur:2,continuous_bernoulli:21,convolution_block:40,crop:51,data:2,datafram:[1,10,11,12,13,14,15,16,17,18,48,49],dataframe_dataset:11,datamodul:[7,8],debug:0,from:4,get:2,group:2,imag:[16,19,50,51,52,53,54,55],image_va:34,instal:4,intro:3,io:[9,10,11,12,13,14,15,16,17,18,19],kl_diverg:22,layer:[39,40,41,42],load:2,loader:[12,13,14,15,16,17],loss:[20,21,22],make:2,manifest_datamodul:8,manifestdatamodul:2,ml:0,ml_op:[23,24,25,26],mlflow:2,mlflow_util:25,mlp:[43,44],model:[2,27,28,29,30,31,32,33,34,35],modul:[8,11,13,14,15,16,17,18,19,21,22,24,25,26,28,29,31,33,34,35,38,40,41,42,44,46,49,51,52,53,54,55],multipl:1,network:[36,37,38,39,40,41,42,43,44,45,46],normal:52,oper:0,optimizer_util:31,own:2,packag:[6,7,9,10,12,20,23,27,30,32,36,37,39,43,45,47,48,50],pad:53,pipelin:1,predict:[0,2],project:[2,54],random_imag:17,reader:18,releas:4,serotini:[0,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],singl:1,skip_connect:41,sourc:4,spatial_pyramid_pool:42,stabl:4,start:2,structur:2,submodul:[7,9,10,12,20,23,27,30,32,37,39,43,45,48,50],subpackag:[6,9,10,27,36,47],swap:55,tabular_va:35,test:[0,1],train:[0,2],trainer:2,transform:[1,47,48,49,50,51,52,53,54,55],us:2,util:[26,30,31,45,46],vae:[32,33,34,35],weight_init:46,wrangl:1,your:2}})