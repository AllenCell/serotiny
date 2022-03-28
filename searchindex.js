Search.setIndex({docnames:["cli","dataframe_transforms","getting_started","index","installation","modules","serotiny","serotiny.datamodules","serotiny.datamodules.manifest_datamodule","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.dataframe_dataset","serotiny.io.dataframe.loaders","serotiny.io.dataframe.loaders.abstract_loader","serotiny.io.dataframe.loaders.classes","serotiny.io.dataframe.loaders.columns","serotiny.io.dataframe.loaders.image","serotiny.io.dataframe.loaders.random_image","serotiny.io.dataframe.readers","serotiny.io.image","serotiny.losses","serotiny.losses.continuous_bernoulli","serotiny.losses.kl_divergence","serotiny.ml_ops","serotiny.ml_ops.ml_ops","serotiny.ml_ops.mlflow_utils","serotiny.ml_ops.utils","serotiny.models","serotiny.models.base_model","serotiny.models.utils","serotiny.models.utils.optimizer_utils","serotiny.models.vae","serotiny.models.vae.base_vae","serotiny.models.vae.image_vae","serotiny.models.vae.tabular_vae","serotiny.networks","serotiny.networks.basic_cnn","serotiny.networks.basic_cnn.basic_cnn","serotiny.networks.layers","serotiny.networks.layers.convolution_block","serotiny.networks.layers.skip_connection","serotiny.networks.layers.spatial_pyramid_pool","serotiny.networks.mlp","serotiny.networks.mlp.mlp","serotiny.networks.utils","serotiny.networks.utils.weight_init","serotiny.transforms","serotiny.transforms.dataframe","serotiny.transforms.dataframe.transforms","serotiny.transforms.image","serotiny.transforms.image.crop","serotiny.transforms.image.normalize","serotiny.transforms.image.pad","serotiny.transforms.image.project","serotiny.transforms.image.swap"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","dataframe_transforms.rst","getting_started.rst","index.rst","installation.rst","modules.rst","serotiny.rst","serotiny.datamodules.rst","serotiny.datamodules.manifest_datamodule.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.dataframe_dataset.rst","serotiny.io.dataframe.loaders.rst","serotiny.io.dataframe.loaders.abstract_loader.rst","serotiny.io.dataframe.loaders.classes.rst","serotiny.io.dataframe.loaders.columns.rst","serotiny.io.dataframe.loaders.image.rst","serotiny.io.dataframe.loaders.random_image.rst","serotiny.io.dataframe.readers.rst","serotiny.io.image.rst","serotiny.losses.rst","serotiny.losses.continuous_bernoulli.rst","serotiny.losses.kl_divergence.rst","serotiny.ml_ops.rst","serotiny.ml_ops.ml_ops.rst","serotiny.ml_ops.mlflow_utils.rst","serotiny.ml_ops.utils.rst","serotiny.models.rst","serotiny.models.base_model.rst","serotiny.models.utils.rst","serotiny.models.utils.optimizer_utils.rst","serotiny.models.vae.rst","serotiny.models.vae.base_vae.rst","serotiny.models.vae.image_vae.rst","serotiny.models.vae.tabular_vae.rst","serotiny.networks.rst","serotiny.networks.basic_cnn.rst","serotiny.networks.basic_cnn.basic_cnn.rst","serotiny.networks.layers.rst","serotiny.networks.layers.convolution_block.rst","serotiny.networks.layers.skip_connection.rst","serotiny.networks.layers.spatial_pyramid_pool.rst","serotiny.networks.mlp.rst","serotiny.networks.mlp.mlp.rst","serotiny.networks.utils.rst","serotiny.networks.utils.weight_init.rst","serotiny.transforms.rst","serotiny.transforms.dataframe.rst","serotiny.transforms.dataframe.transforms.rst","serotiny.transforms.image.rst","serotiny.transforms.image.crop.rst","serotiny.transforms.image.normalize.rst","serotiny.transforms.image.pad.rst","serotiny.transforms.image.project.rst","serotiny.transforms.image.swap.rst"],objects:{"":[[6,0,0,"-","serotiny"]],"serotiny.datamodules":[[7,1,1,"","ManifestDatamodule"],[8,0,0,"-","manifest_datamodule"]],"serotiny.datamodules.ManifestDatamodule":[[7,2,1,"","make_dataloader"],[7,2,1,"","predict_dataloader"],[7,2,1,"","test_dataloader"],[7,2,1,"","train_dataloader"],[7,2,1,"","val_dataloader"]],"serotiny.datamodules.manifest_datamodule":[[8,1,1,"","ManifestDatamodule"]],"serotiny.datamodules.manifest_datamodule.ManifestDatamodule":[[8,2,1,"","make_dataloader"],[8,2,1,"","predict_dataloader"],[8,2,1,"","test_dataloader"],[8,2,1,"","train_dataloader"],[8,2,1,"","val_dataloader"]],"serotiny.io":[[10,0,0,"-","dataframe"],[19,0,0,"-","image"]],"serotiny.io.dataframe":[[11,0,0,"-","dataframe_dataset"],[12,0,0,"-","loaders"],[18,0,0,"-","readers"]],"serotiny.io.dataframe.dataframe_dataset":[[11,1,1,"","DataframeDataset"]],"serotiny.io.dataframe.loaders":[[12,1,1,"","LoadClass"],[12,1,1,"","LoadColumn"],[12,1,1,"","LoadColumns"],[12,1,1,"","LoadImage"],[13,0,0,"-","abstract_loader"],[14,0,0,"-","classes"],[15,0,0,"-","columns"],[16,0,0,"-","image"],[17,0,0,"-","random_image"]],"serotiny.io.dataframe.loaders.abstract_loader":[[13,1,1,"","Loader"]],"serotiny.io.dataframe.loaders.classes":[[14,1,1,"","LoadClass"]],"serotiny.io.dataframe.loaders.columns":[[15,1,1,"","LoadColumn"],[15,1,1,"","LoadColumns"]],"serotiny.io.dataframe.loaders.image":[[16,1,1,"","LoadImage"]],"serotiny.io.dataframe.loaders.random_image":[[17,1,1,"","LoadRandomTensor"]],"serotiny.io.dataframe.readers":[[18,3,1,"","filter_columns"],[18,3,1,"","read_csv"],[18,3,1,"","read_dataframe"],[18,3,1,"","read_parquet"]],"serotiny.io.image":[[19,3,1,"","image_loader"],[19,3,1,"","infer_dims"],[19,3,1,"","tiff_writer"]],"serotiny.losses":[[21,0,0,"-","continuous_bernoulli"],[22,0,0,"-","kl_divergence"]],"serotiny.losses.continuous_bernoulli":[[21,1,1,"","CBLogLoss"]],"serotiny.losses.continuous_bernoulli.CBLogLoss":[[21,2,1,"","forward"],[21,4,1,"","reduction"]],"serotiny.losses.kl_divergence":[[22,3,1,"","diagonal_gaussian_kl"],[22,3,1,"","isotropic_gaussian_kl"]],"serotiny.ml_ops":[[24,0,0,"-","ml_ops"]],"serotiny.models":[[28,0,0,"-","base_model"],[29,0,0,"-","utils"],[31,0,0,"-","vae"]],"serotiny.models.base_model":[[28,1,1,"","BaseModel"]],"serotiny.models.base_model.BaseModel":[[28,2,1,"","configure_optimizers"],[28,2,1,"","forward"],[28,2,1,"","parse_batch"],[28,2,1,"","test_epoch_end"],[28,2,1,"","test_step"],[28,2,1,"","train_epoch_end"],[28,4,1,"","training"],[28,2,1,"","training_step"],[28,2,1,"","validation_epoch_end"],[28,2,1,"","validation_step"]],"serotiny.models.utils":[[30,0,0,"-","optimizer_utils"]],"serotiny.models.utils.optimizer_utils":[[30,3,1,"","find_lr_scheduler"],[30,3,1,"","find_optimizer"]],"serotiny.models.vae":[[32,0,0,"-","base_vae"],[33,0,0,"-","image_vae"],[34,0,0,"-","tabular_vae"]],"serotiny.models.vae.base_vae":[[32,1,1,"","BaseVAE"]],"serotiny.models.vae.base_vae.BaseVAE":[[32,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[32,2,1,"","calculate_elbo"],[32,2,1,"","forward"],[32,2,1,"","parse_batch"],[32,4,1,"","precision"],[32,4,1,"","prepare_data_per_node"],[32,2,1,"","sample_z"],[32,4,1,"","training"],[32,4,1,"","use_amp"]],"serotiny.models.vae.image_vae":[[33,1,1,"","ImageVAE"]],"serotiny.models.vae.image_vae.ImageVAE":[[33,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[33,4,1,"","precision"],[33,4,1,"","prepare_data_per_node"],[33,4,1,"","training"],[33,4,1,"","use_amp"]],"serotiny.models.vae.tabular_vae":[[34,1,1,"","TabularVAE"]],"serotiny.models.vae.tabular_vae.TabularVAE":[[34,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[34,4,1,"","precision"],[34,4,1,"","prepare_data_per_node"],[34,4,1,"","training"],[34,4,1,"","use_amp"]],"serotiny.networks":[[36,0,0,"-","basic_cnn"],[38,0,0,"-","layers"],[42,0,0,"-","mlp"],[44,0,0,"-","utils"]],"serotiny.networks.basic_cnn":[[37,0,0,"-","basic_cnn"]],"serotiny.networks.basic_cnn.basic_cnn":[[37,1,1,"","BasicCNN"]],"serotiny.networks.basic_cnn.basic_cnn.BasicCNN":[[37,2,1,"","conv_forward"],[37,2,1,"","forward"],[37,4,1,"","training"]],"serotiny.networks.layers":[[39,0,0,"-","convolution_block"],[40,0,0,"-","skip_connection"],[41,0,0,"-","spatial_pyramid_pool"]],"serotiny.networks.layers.convolution_block":[[39,1,1,"","ConvBlock"],[39,3,1,"","conv_block"]],"serotiny.networks.layers.convolution_block.ConvBlock":[[39,2,1,"","forward"],[39,4,1,"","training"]],"serotiny.networks.layers.skip_connection":[[40,1,1,"","SkipConnection"]],"serotiny.networks.layers.skip_connection.SkipConnection":[[40,2,1,"","forward"],[40,4,1,"","training"]],"serotiny.networks.layers.spatial_pyramid_pool":[[41,1,1,"","SpatialPyramidPool"],[41,3,1,"","spatial_pyramid_pool"]],"serotiny.networks.layers.spatial_pyramid_pool.SpatialPyramidPool":[[41,2,1,"","forward"],[41,4,1,"","training"]],"serotiny.networks.mlp":[[43,0,0,"-","mlp"]],"serotiny.networks.mlp.mlp":[[43,1,1,"","MLP"]],"serotiny.networks.mlp.mlp.MLP":[[43,2,1,"","forward"],[43,4,1,"","training"]],"serotiny.networks.utils":[[45,0,0,"-","weight_init"]],"serotiny.networks.utils.weight_init":[[45,3,1,"","weight_init"]],"serotiny.transforms":[[47,0,0,"-","dataframe"],[49,0,0,"-","image"]],"serotiny.transforms.dataframe":[[48,0,0,"-","transforms"]],"serotiny.transforms.dataframe.transforms":[[48,3,1,"","append_class_weights"],[48,3,1,"","append_labels_to_integers"],[48,3,1,"","append_one_hot"],[48,3,1,"","filter_columns"],[48,3,1,"","filter_rows"],[48,3,1,"","make_random_df"],[48,3,1,"","sample_n_each"],[48,3,1,"","split_dataframe"]],"serotiny.transforms.image":[[50,0,0,"-","crop"],[51,0,0,"-","normalize"],[52,0,0,"-","pad"],[53,0,0,"-","project"],[54,0,0,"-","swap"]],"serotiny.transforms.image.crop":[[50,1,1,"","CropCenter"]],"serotiny.transforms.image.normalize":[[51,1,1,"","NormalizeAbsolute"],[51,1,1,"","NormalizeMean"],[51,1,1,"","NormalizeMinMax"]],"serotiny.transforms.image.pad":[[52,1,1,"","ExpandColumns"],[52,1,1,"","ExpandTo"],[52,1,1,"","PadTo"],[52,3,1,"","expand_columns"],[52,3,1,"","expand_to"],[52,3,1,"","pull_to"],[52,3,1,"","split_number"],[52,3,1,"","to_tensor"]],"serotiny.transforms.image.project":[[53,1,1,"","Project"]],"serotiny.transforms.image.swap":[[54,1,1,"","Permute"],[54,1,1,"","SwapAxes"]],serotiny:[[7,0,0,"-","datamodules"],[9,0,0,"-","io"],[20,0,0,"-","losses"],[27,0,0,"-","models"],[35,0,0,"-","networks"],[46,0,0,"-","transforms"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[1,2,22,33,34,39,50,51,52],"06845":21,"1":[1,2,22,33,34,48,51],"10":2,"100":[2,48],"11":0,"1234":2,"13":0,"1337":2,"16":2,"1907":21,"1e":2,"2":[1,2,51],"256":43,"2d":39,"3":[1,2,37,39],"32":2,"3d":[33,37,39],"4":1,"42":48,"5":2,"50":[0,1],"64":2,"7":1,"abstract":13,"boolean":32,"case":[2,7,8,48],"class":[2,7,8,10,11,12,13,15,16,17,21,28,32,33,34,37,39,40,41,43,48,50,51,52,53,54],"default":[0,2,39,48],"do":[0,2],"final":2,"float":[12,15,32,33,34,48],"function":[0,2,11,32,39,48],"import":2,"int":[12,14,17,19,32,33,34,37,39,48,52,54],"public":4,"return":[12,14,18,19,48],"true":[2,12,16,19,33,37,39,48,50],"try":2,A:[2,7,8,11,12,15,18,48],AND:18,And:37,As:2,But:[12,16],By:48,For:[0,1,2,12,16,18],If:[2,4,7,8,12,15,18,19,48],In:[0,2,11,48],It:[1,3,7,8,12,15,17,22,48],On:2,One:2,Or:4,That:[0,1,2],The:[0,4,11,12,15,19,48],There:2,These:[0,2],To:[1,2,4],With:2,_:2,_loss:[21,32,33],_partial_:2,_target_:2,a_datamodule_config:0,a_model_config:0,a_tiny_project:2,ab:21,abl:2,about:[0,2],abov:[0,2],abstract_load:[10,12,14,15,16,17],ad:2,adam:[2,28,32,33,34],add:[0,2,48],addit:[0,7,8],addition:2,ahead:2,aic:11,aics_imag:19,aicsimag:19,aicsimageio:[12,16,19],all:[1,2,18,48],allencellmodel:[2,4],allow:1,allow_zero_length_dataloader_with_multiple_devic:[32,33,34],also:[0,1,18,22],altern:2,alwai:[2,4,13],among:[7,8],an:[0,2,12,15,19,22],ani:[0,2,7,8],anisotrop:32,append:[0,48],append_class_weight:48,append_labels_to_integ:48,append_one_hot:48,appli:[12,16,18,19,37,39,45],applic:2,appropri:[2,11],ar:[0,2,7,8,12,13,15,18,48],architectur:2,aren:[0,18],arg:[7,8],argument:[0,1,2,7,8,18,48],arrai:[12,15,16,19,32,33,34,52],artifact:[2,3],arxiv:21,asid:[7,8],assert:18,assign:19,assum:2,astyp:[12,16],avail:2,ax:52,axi:[51,53],b:[1,2,48],backend:2,balanc:48,base:[7,8,11,12,13,14,15,16,17,21,28,32,33,34,37,39,40,41,43,48,50,51,52,53,54],base_model:[6,27,32],base_va:[27,31,33,34],basemodel:[28,32],baseva:[32,33,34],bash:0,basic:32,basic_cnn:[6,35],basiccnn:37,batch:[22,28,32],batch_idx:28,batch_norm:39,batch_siz:[2,7,8],becaus:0,becom:2,befor:19,behavior:0,behind:0,belong:48,bernoulli:21,beta:[32,33,34],better:2,between:[22,48],binari:[12,14],block:[2,39],bolt:2,bool:[12,14,16,19,28,32,33,34,37,39,40,41,43,48],build:2,c:[1,48],cache_output:[28,32,33,34],calculate_elbo:32,call:[0,2,48],callabl:[12,13,16,19],calll:2,can:[0,1,2,4,7,8,32,39,48],cblogloss:21,cell:[0,11],cellid:1,center_of_mass:50,chang:[0,2],channel:[12,16,19,37,39],channel_nam:19,check:2,cli:[1,2],clip_max:51,clip_min:51,clip_quantil:51,clone:4,cnn:37,collect:2,column:[1,7,8,10,11,12,14,16,17,18,48,52],columns_to_filt:18,com:[2,4],combin:18,come:0,command:[0,2,4],common:1,commun:2,complet:0,comput:[2,22],compute_loss:32,concaten:[12,15],condit:18,config:0,configur:[0,3],configure_optim:28,consist:[11,17],constant:52,construct:2,consum:2,contain:[2,7,8,11,12,14,15,16,18,19,48],content:11,context:0,continu:21,continuous_bernoulli:[6,20],contribut:2,conv_block:39,conv_forward:37,convblock:39,convert:48,convolut:39,convolution_block:[35,38],cookiecutt:2,copi:4,core:[7,8,18,28,48],correct:19,correspond:[2,11,19,48],could:2,coupl:2,covari:32,creat:[0,1,2,7,8,48],crop:[46,49],crop_raw:1,crop_seg:1,cropcent:50,cropi:50,cropx:50,cropz:50,csv:[1,2,18],curl:4,current:32,custom:2,d:48,dashboard:2,data:[0,7,8,11,13,39,48],datafram:[0,2,6,9,46],dataframe_dataset:[9,10],dataframedataset:[11,13],dataload:[7,8],dataloader_kwarg:[7,8],datamodul:[2,5,6],dataset:[2,7,8,11,48],decod:32,decoder_non_linear:33,deep:3,defin:1,depth:37,describ:2,desir:18,destin:0,detail:48,determin:[12,14,19,32],develop:2,diagon:[22,32],diagonal_gaussian_kl:22,dict:[7,8,11],differ:[0,2,18,48],dim:[17,43],dim_ord:19,dimens:[17,19,37,39,52],dimension:[37,39],directli:[2,12,15],disk:19,diverg:22,doc:[2,7,8,12,16],doesn:[2,19],don:4,done:[0,1],download:4,drive:3,dtype:[12,15,16,19],dummi:17,dure:2,e:[0,2,18,32],each:[0,2,7,8,11,19,37,48],earli:2,earlystop:2,easier:2,either:[2,4,7,8,18,32,48],els:2,enabl:[12,15,48],enable_checkpoint:2,encod:[32,48],end:[1,2,12,15,18,48],endswith:[12,15,18,48],equal:48,etc:[2,7,8],eval:[0,2],evalu:0,everyth:[7,8],exampl:[0,2],exclud:[1,12,15,18,48],execut:3,exist:48,expand_column:52,expand_to:52,expandcolumn:52,expanded_column:52,expandto:52,expect:[2,18],experi:[1,2,3],experiment_nam:2,explicit:[12,15,48],expos:1,express:[12,15,18,48],extend:48,extract:13,fact:2,fail:2,fals:[1,12,14,19,32,33,34,37,39,48,51],familiar:2,fashion:2,feature_:2,field:[12,15],file:[0,1,2,7,8,11,12,16,18],file_typ:[12,16],filter:[1,12,15,18,48],filter_column:[1,12,15,18,48],filter_row:[1,48],find_lr_schedul:30,find_optim:30,first:[1,22,54],fix:32,flag:[0,12,14,19,32],flat_output:37,float32:[12,16,19],fold:[7,8],folder:2,follow:2,forc:48,force_s:50,form:2,format:[12,16],forward:[21,28,32,37,39,40,41,43],found:18,fraction:48,frame:[18,48],framework:[2,3],from:[0,2,7,8,11,12,13,14,15,16,18,19,30,32],futur:[12,16],g:[0,2],gaussian:22,gener:[1,2,12,16,17,48],get:[0,30],git:4,github:[2,4],given:[0,2,7,8,12,16,18,19,30,48],go:2,govern:2,group:0,guid:4,ha:[0,2,48],half:48,have:[0,2,4,19,48],heavili:2,help:0,helper:1,here:[0,1,2,21],hidden:37,hidden_channel:[33,37],hidden_lay:[2,34,43],hood:0,hot:[12,14,48],how:0,html:[7,8,12,16],http:[2,4,7,8,12,16,21],hydra:[0,2,3],hyperparamet:2,i:[18,32],id:2,id_label:[28,32,33,34],identifi:[2,48],ignor:[12,15,17,18,48],imag:[0,6,9,10,11,12,46],image_load:19,image_va:[27,31],imageva:33,img:19,implement:[1,2,13],implicitli:37,importantli:1,in_c:39,in_channel:[33,37],includ:[12,15,16,18,48],include_column:18,incomplet:2,independ:2,index:19,infer:[0,19],infer_dim:19,info:[0,2,7,8,12,16],inform:0,initi:[0,32],inner:2,input:[1,21,37,39,48],input_dim:[33,37],input_s:2,instal:[0,2],instanc:13,instanti:[2,7,8,32,37,39],instead:2,integ:[19,48],integr:3,intend:0,interact:[0,2],interfac:[2,32],interpol:2,intimid:0,involv:1,io:[2,5,6],ipynb:0,isotrop:[22,32,33,34],isotropic_gaussian_kl:22,its:[0,2,13,48],jupyt:2,just:[0,2],keep:48,kei:[2,11],kept:17,kernel:39,kernel_s:[37,39],keyword:[7,8],kl_diverg:[6,20],kld:32,know:1,kullback:22,kwarg:[7,8,28,32],label:[19,32],latenc:18,latent_dim:[0,2,32,33,34],later:2,launcher:0,layer:[6,35,37],learn:[0,2,3,32],learn_prior_logvar:[32,33,34],leibler:22,less:48,let:1,level:2,leverag:[0,2,12,15],lightiningdatamodul:2,lightn:[2,3,7,8,28],lightningdatamodul:[7,8],lightningmodul:[2,28],like:[0,2,12,16],line:[0,2],list:[2,7,8,12,15,16,18,19,48],ll:0,load:[7,8,11,12,13,16,17,18,19],load_as_torch:[12,16],load_model_from_checkpoint:2,loadclass:[12,14],loadcolumn:[2,12,15],loader:[2,7,8,9,10,11],loadimag:[12,16],loadrandomtensor:17,log:[22,32],log_var:22,logic:[2,18],logvar1:22,logvar2:22,logvar:[22,32],loop:2,loss:[5,6,32,33],loss_mask_label:[32,33,34],lr:2,lr_schedul:30,m:[0,45],machin:2,made:[7,8],mai:2,main:[2,4],make:[0,2],make_dataload:[7,8],make_notebook:0,make_random_df:[1,48],manag:3,mani:0,manifest:[1,2,7,8,11,13],manifest_datamodul:[6,7],manifestdatamodul:[2,7,8],map:19,mask:32,match:[12,15,18,48],matrix:32,max:[37,53],max_epoch:[0,2],max_pool_lay:[33,37],mean:[0,22,33],mechan:13,merg:1,method:[4,11],metric:2,might:[2,12,16],mimic:0,min_delta:2,ml:2,ml_op:[2,5,6],mlflow:[0,3],mlflow_util:[2,6,23],mlp:[6,35],mode:[21,33,37,39,52,53],model:[0,5,6,45],modifi:48,modul:[0,1,2,6,7,9,10,12,20,23,27,29,31,35,36,38,42,44,46,47,49],modular:2,modularli:2,monitor:2,more:[0,2,7,8,12,16,48],more_param:2,most:4,mostli:1,mseloss:[32,33],mu1:22,mu2:22,mu:[22,32],multi:0,multipl:[0,12,15],must:[12,15,18,19,48],my:2,my_input:1,my_output:1,mycustommodel:2,n:52,n_row:[1,48],name:[2,7,8,12,14,18,19,30],ndarrai:[12,16],necessarili:22,need:1,net:3,network:[5,6,32],neural:3,newli:2,nn:[21,32,33,37,39,40,41,43],non_linear:[33,37,39],none:[7,8,12,15,16,18,19,32,33,34,37,39,43,48,50,51,52],nonetyp:28,noqa:[12,16],normal:[11,46,49],normalizeabsolut:51,normalizemean:51,normalizeminmax:51,note:[0,1,2],notebook:[0,2],notion:2,now:[0,2,12,16],np:[12,16,19],num_class:[12,14],num_work:[2,7,8],number:[12,14,16,19,37,39,48],numpi:[12,16,19,32,33,34],object:[2,13,50,51,52,53,54],obtain:0,ol:4,ome_tiff_read:[12,16],omegaconf:2,ometiffread:[12,16],omit:[0,2],onc:[0,4],one:[0,2,7,8,12,14,48],one_hot_encod:48,onli:[0,2,7,8,12,16,18,19,32,48],oper:[2,37,48],optim:[2,28,30,32,33,34],optimizer_nam:30,optimizer_util:[27,29],option:[0,7,8,12,15,16,18,19,32,33,34,37,39,48],order:[19,51],org:[7,8,12,16,21],organ:2,other:[2,12,16,17],otherwis:18,our:2,out:2,out_c:39,out_pool_s:41,output:[19,28,37,39],output_dim:37,output_dtyp:19,output_path:1,over:2,overrid:[0,2],own:2,p:2,packag:[2,5],package_nam:2,pad:[39,46,49,50],padto:52,panda:[11,13,18,48],param1:2,param2:2,paramet:[0,2,7,8,11,12,14,15,16,17,18,19,22,32,37,39,48],parquet:[7,8,18],parse_batch:[28,32],part:2,particular:0,partit:1,pass:[7,8],path:[0,2,7,8,11,12,16,18,19],pathlib:[18,19],patienc:2,pattern:18,pd:[11,18,48],per:48,perform:48,permut:54,pip:[2,4],pipelin:48,place:2,placehold:1,point:[1,11],pool:37,popul:0,port:2,possibl:[2,12,14],practic:11,preced:[1,18],precis:[32,33,34],predict:[2,3],predict_dataload:[7,8],prefer:4,prepare_data_per_nod:[32,33,34],previou:1,prior:32,prior_logvar:[32,33,34],prior_mod:[32,33,34],prob:21,process:[2,4],programat:2,project:[0,3,46,49],project_descript:2,project_nam:2,propos:21,provid:[1,2,13,48],pull_to:52,purpos:2,py:[2,4],pyramid_pool_split:37,python:[2,4],pytorch:[2,3,7,8,11,32],pytorch_lightn:[2,7,8,28],queri:[12,15,18],random:[1,48],random_imag:[10,12],randomli:17,rather:[12,16],re:[0,1],read:[2,18],read_csv:18,read_datafram:18,read_parquet:18,reader:[9,10,12,16,19],readi:2,reason:[0,2],recent:4,recogn:2,recommend:2,reconstruct:32,reconstruction_loss:[32,33],reconstruction_reduc:[32,33],reduct:21,refer:[2,12,16],regex:[12,15,18,48],regular:[12,15,18,48],relat:2,reli:2,remain:[2,12,15,18,48],rememb:2,replac:48,repo:4,repositori:4,requir:[2,12,16,18],required_column:18,respect:[0,18,32],restart:2,result:[1,2,12,15,18,19,48],retriev:[12,14,15,16,19,32],return_as_torch:19,return_channel:19,return_s:37,return_split:[1,48],return_torch:51,right:2,row:[1,13,48,52],run:[0,2,4],run_nam:2,s:[0,2,7,8,48],same:[2,32],sampl:48,sample_n_each:48,sample_z:32,save:19,sc:0,scale:51,scenario:2,schedul:30,scheduler_nam:30,script:0,second:[1,22,54],section:0,see:[0,2,7,8,12,16,48],seed:[0,2,48],select:[2,12,15,48],select_channel:[12,16,19],separ:48,sequenc:[7,8,12,15,16,17,18,19,28,32,33,34,37,39,48],serotini:[1,2,4],serotiny_bash:0,server:2,session:0,set:[2,3],setup:[2,4],shall:13,shell:0,should:2,shuffl:[7,8],signifi:1,simpl:[2,12,15],singl:[7,8],skip_connect:[33,35,37,38,39],skipconnect:40,slower:18,smaller:48,so:[0,2],some:[2,12,16],someon:2,someth:2,sourc:[7,8,11,12,13,14,15,16,17,18,19,21,22,28,30,32,33,34,37,39,40,41,43,45,48,50,51,52,53,54],spatial_pyramid_pool:[35,38],spatialpyramidpool:41,specif:[2,7,8,13],specifi:[0,2],split:[1,2,7,8,48],split_column:[2,7,8],split_datafram:[1,48],split_numb:52,stabl:[7,8,12,16],stand:2,standard:2,start:[0,1,12,15,18,48],startswith:[2,12,15,18,48],step:1,stop:2,store:[2,18],str:[7,8,12,14,15,16,17,18,19,21,28,32,33,34,37,39,48],straightforward:0,stratifi:48,string:[2,12,15,18,32,48],structur:3,subclass:[2,13,32],submodul:[5,6,35,46],subpackag:5,substr:[12,15,18,48],sum:[21,32],suppli:[12,15,18,48],support:[12,16,18,32],swap:[2,46,49],swapax:54,sweep:[0,2],syntax:[0,2],t:[0,2,4,18,19],tab:0,tabular_va:[27,31],tabularva:[2,34],take:18,tarbal:4,target:21,target_dim:[52,54],templat:2,tensor:[12,16,17,19,22,32,33,34],term:32,termin:4,test:[2,3,7,8,28,32,33,34,48],test_dataload:[7,8],test_epoch_end:28,test_step:28,than:[12,16,48],the_run_id:2,the_tracking_uri:2,thei:2,them:[0,1,2,12,13,15],thi:[0,1,2,4,11,12,13,16,18,19,48],third:1,three:[2,7,8,48],through:4,throughout:2,thrown:18,thu:2,tiff:[12,16,19],tiff_writ:19,tini:2,tinker:0,to_tensor:52,todo:48,toggl:48,too:[2,12,16],tool:[1,3],top:2,torch:[2,7,8,11,12,16,19,21,22,28,30,32,33,34,37,39,40,41,43],track:3,tracking_uri:2,train:[1,3,7,8,28,32,33,34,37,39,40,41,43,48],train_dataload:[7,8],train_epoch_end:28,train_frac:[1,48],trainer:0,training_step:28,transform:[5,6,12,16,19],trivial:2,two:22,type:[12,16,18,19,28,32,34],under:[0,2],underli:17,understand:[0,2],union:[7,8,12,16,18,19,32,33,34,37,48],uniqu:[2,48],unlik:2,up:2,up_conv:[37,39],upon:[12,16],upsampl:48,upsample_lay:37,uri:2,us:[0,1,2,7,8,11,12,13,14,15,16,17,18,19,32,48],usag:[2,45],use_amp:[32,33,34],user:1,util:[0,6,7,8,11,23,27,35,39],vae:[6,27],val:[1,7,8,48],val_dataload:[7,8],val_frac:48,val_loss:2,valid:[2,48],validation_epoch_end:28,validation_step:28,valu:[0,2,12,14,32,39,48,52],valueerror:18,varianc:[22,32],ve:0,vector:[12,14],via:2,wai:2,want:[0,1,2],we:[1,2,11],weight:[32,48],weight_init:[35,44],well:[0,2],what:[0,2],when:[0,2,7,8,19],where:[2,19,48],wherea:2,whether:[12,14,16,19,32,48],which:[0,1,2,7,8,11,12,13,16,32,37,48],wise:22,within:2,work:[2,22],would:2,wrangl:[0,48],wrap:11,wrapped_modul:40,write:[2,19],x1:43,x2:43,x:[2,28,32,37,39,40,41],x_dim:[2,34],x_hat:32,x_label:[2,28,32,33,34],y_encoded_label:[12,14],yaml:2,you:[0,1,2,4],your:[0,2,4],your_data_config_nam:2,your_model_config_nam:2,yourcustomcallback:2,zarr:[12,16]},titles:["serotiny CLI","Dataframe wrangling","Getting started","serotiny","Installation","serotiny","serotiny package","serotiny.datamodules package","serotiny.datamodules.manifest_datamodule module","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.dataframe_dataset module","serotiny.io.dataframe.loaders package","serotiny.io.dataframe.loaders.abstract_loader module","serotiny.io.dataframe.loaders.classes module","serotiny.io.dataframe.loaders.columns module","serotiny.io.dataframe.loaders.image module","serotiny.io.dataframe.loaders.random_image module","serotiny.io.dataframe.readers module","serotiny.io.image module","serotiny.losses package","serotiny.losses.continuous_bernoulli module","serotiny.losses.kl_divergence module","serotiny.ml_ops package","serotiny.ml_ops.ml_ops module","serotiny.ml_ops.mlflow_utils module","serotiny.ml_ops.utils module","serotiny.models package","serotiny.models.base_model module","serotiny.models.utils package","serotiny.models.utils.optimizer_utils module","serotiny.models.vae package","serotiny.models.vae.base_vae module","serotiny.models.vae.image_vae module","serotiny.models.vae.tabular_vae module","serotiny.networks package","serotiny.networks.basic_cnn package","serotiny.networks.basic_cnn.basic_cnn module","serotiny.networks.layers package","serotiny.networks.layers.convolution_block module","serotiny.networks.layers.skip_connection module","serotiny.networks.layers.spatial_pyramid_pool module","serotiny.networks.mlp package","serotiny.networks.mlp.mlp module","serotiny.networks.utils package","serotiny.networks.utils.weight_init module","serotiny.transforms package","serotiny.transforms.dataframe package","serotiny.transforms.dataframe.transforms module","serotiny.transforms.image package","serotiny.transforms.image.crop module","serotiny.transforms.image.normalize module","serotiny.transforms.image.pad module","serotiny.transforms.image.project module","serotiny.transforms.image.swap module"],titleterms:{"class":14,The:2,abstract_load:13,base_model:28,base_va:32,basic_cnn:[36,37],call:1,callback:2,chain:1,cli:0,column:15,config:2,configur:2,continuous_bernoulli:21,convolution_block:39,crop:50,data:2,datafram:[1,10,11,12,13,14,15,16,17,18,47,48],dataframe_dataset:11,datamodul:[7,8],debug:0,from:4,get:2,group:2,imag:[16,19,49,50,51,52,53,54],image_va:33,instal:4,io:[9,10,11,12,13,14,15,16,17,18,19],kl_diverg:22,layer:[38,39,40,41],load:2,loader:[12,13,14,15,16,17],loss:[20,21,22],manifest_datamodul:8,ml:0,ml_op:[23,24,25,26],mlflow:2,mlflow_util:25,mlp:[42,43],model:[2,27,28,29,30,31,32,33,34],modul:[8,11,13,14,15,16,17,18,19,21,22,24,25,26,28,30,32,33,34,37,39,40,41,43,45,48,50,51,52,53,54],multipl:1,network:[35,36,37,38,39,40,41,42,43,44,45],normal:51,oper:0,optimizer_util:30,packag:[6,7,9,10,12,20,23,27,29,31,35,36,38,42,44,46,47,49],pad:52,pipelin:1,predict:0,project:[2,53],random_imag:17,reader:18,releas:4,serotini:[0,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],singl:1,skip_connect:40,sourc:4,spatial_pyramid_pool:41,stabl:4,start:2,structur:2,submodul:[7,9,10,12,20,23,27,29,31,36,38,42,44,47,49],subpackag:[6,9,10,27,35,46],swap:54,tabular_va:34,test:[0,1],train:[0,2],trainer:2,transform:[1,46,47,48,49,50,51,52,53,54],util:[26,29,30,44,45],vae:[31,32,33,34],weight_init:45,wrangl:1}})