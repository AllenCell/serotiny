Search.setIndex({docnames:["cli","dataframe_transforms","getting_started","index","installation","modules","serotiny","serotiny.datamodules","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.loaders","serotiny.losses","serotiny.ml_ops","serotiny.models","serotiny.models.utils","serotiny.models.vae","serotiny.networks","serotiny.networks.basic_cnn","serotiny.networks.layers","serotiny.networks.mlp","serotiny.networks.utils","serotiny.transforms","serotiny.transforms.dataframe","serotiny.transforms.image"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","dataframe_transforms.rst","getting_started.rst","index.rst","installation.rst","modules.rst","serotiny.rst","serotiny.datamodules.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.loaders.rst","serotiny.losses.rst","serotiny.ml_ops.rst","serotiny.models.rst","serotiny.models.utils.rst","serotiny.models.vae.rst","serotiny.networks.rst","serotiny.networks.basic_cnn.rst","serotiny.networks.layers.rst","serotiny.networks.mlp.rst","serotiny.networks.utils.rst","serotiny.transforms.rst","serotiny.transforms.dataframe.rst","serotiny.transforms.image.rst"],objects:{"":[[6,0,0,"-","serotiny"]],"serotiny.datamodules":[[7,1,1,"","ManifestDatamodule"],[7,0,0,"-","manifest_datamodule"]],"serotiny.datamodules.ManifestDatamodule":[[7,2,1,"","make_dataloader"],[7,2,1,"","predict_dataloader"],[7,2,1,"","test_dataloader"],[7,2,1,"","train_dataloader"],[7,2,1,"","val_dataloader"]],"serotiny.datamodules.manifest_datamodule":[[7,1,1,"","ManifestDatamodule"]],"serotiny.datamodules.manifest_datamodule.ManifestDatamodule":[[7,2,1,"","make_dataloader"],[7,2,1,"","predict_dataloader"],[7,2,1,"","test_dataloader"],[7,2,1,"","train_dataloader"],[7,2,1,"","val_dataloader"]],"serotiny.io":[[9,0,0,"-","dataframe"],[8,0,0,"-","image"]],"serotiny.io.dataframe":[[9,0,0,"-","dataframe_dataset"],[10,0,0,"-","loaders"],[9,0,0,"-","readers"]],"serotiny.io.dataframe.dataframe_dataset":[[9,1,1,"","DataframeDataset"]],"serotiny.io.dataframe.loaders":[[10,1,1,"","LoadClass"],[10,1,1,"","LoadColumn"],[10,1,1,"","LoadColumns"],[10,1,1,"","LoadImage"],[10,0,0,"-","abstract_loader"],[10,0,0,"-","classes"],[10,0,0,"-","columns"],[10,0,0,"-","image"],[10,0,0,"-","random_image"]],"serotiny.io.dataframe.loaders.abstract_loader":[[10,1,1,"","Loader"]],"serotiny.io.dataframe.loaders.classes":[[10,1,1,"","LoadClass"]],"serotiny.io.dataframe.loaders.columns":[[10,1,1,"","LoadColumn"],[10,1,1,"","LoadColumns"]],"serotiny.io.dataframe.loaders.image":[[10,1,1,"","LoadImage"]],"serotiny.io.dataframe.loaders.random_image":[[10,1,1,"","LoadRandomTensor"]],"serotiny.io.dataframe.readers":[[9,3,1,"","filter_columns"],[9,3,1,"","read_csv"],[9,3,1,"","read_dataframe"],[9,3,1,"","read_parquet"]],"serotiny.io.image":[[8,3,1,"","image_loader"],[8,3,1,"","infer_dims"],[8,3,1,"","tiff_writer"]],"serotiny.losses":[[11,0,0,"-","continuous_bernoulli"],[11,0,0,"-","kl_divergence"]],"serotiny.losses.continuous_bernoulli":[[11,1,1,"","CBLogLoss"]],"serotiny.losses.continuous_bernoulli.CBLogLoss":[[11,2,1,"","forward"],[11,4,1,"","reduction"]],"serotiny.losses.kl_divergence":[[11,3,1,"","diagonal_gaussian_kl"],[11,3,1,"","isotropic_gaussian_kl"]],"serotiny.models":[[13,0,0,"-","base_model"],[14,0,0,"-","utils"],[15,0,0,"-","vae"]],"serotiny.models.base_model":[[13,1,1,"","BaseModel"]],"serotiny.models.base_model.BaseModel":[[13,2,1,"","configure_optimizers"],[13,2,1,"","forward"],[13,2,1,"","parse_batch"],[13,2,1,"","test_epoch_end"],[13,2,1,"","test_step"],[13,2,1,"","train_epoch_end"],[13,4,1,"","training"],[13,2,1,"","training_step"],[13,2,1,"","validation_epoch_end"],[13,2,1,"","validation_step"]],"serotiny.models.utils":[[14,0,0,"-","optimizer_utils"]],"serotiny.models.utils.optimizer_utils":[[14,3,1,"","find_lr_scheduler"],[14,3,1,"","find_optimizer"]],"serotiny.models.vae":[[15,0,0,"-","base_vae"],[15,0,0,"-","image_vae"],[15,0,0,"-","tabular_vae"]],"serotiny.models.vae.base_vae":[[15,1,1,"","BaseVAE"]],"serotiny.models.vae.base_vae.BaseVAE":[[15,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[15,2,1,"","calculate_elbo"],[15,2,1,"","forward"],[15,2,1,"","parse_batch"],[15,4,1,"","precision"],[15,4,1,"","prepare_data_per_node"],[15,2,1,"","sample_z"],[15,4,1,"","training"],[15,4,1,"","use_amp"]],"serotiny.models.vae.image_vae":[[15,1,1,"","ImageVAE"]],"serotiny.models.vae.image_vae.ImageVAE":[[15,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[15,4,1,"","precision"],[15,4,1,"","prepare_data_per_node"],[15,4,1,"","training"],[15,4,1,"","use_amp"]],"serotiny.models.vae.tabular_vae":[[15,1,1,"","TabularVAE"]],"serotiny.models.vae.tabular_vae.TabularVAE":[[15,4,1,"","allow_zero_length_dataloader_with_multiple_devices"],[15,4,1,"","precision"],[15,4,1,"","prepare_data_per_node"],[15,4,1,"","training"],[15,4,1,"","use_amp"]],"serotiny.networks":[[17,0,0,"-","basic_cnn"],[18,0,0,"-","layers"],[19,0,0,"-","mlp"],[20,0,0,"-","utils"]],"serotiny.networks.basic_cnn":[[17,0,0,"-","basic_cnn"]],"serotiny.networks.basic_cnn.basic_cnn":[[17,1,1,"","BasicCNN"]],"serotiny.networks.basic_cnn.basic_cnn.BasicCNN":[[17,2,1,"","conv_forward"],[17,2,1,"","forward"],[17,4,1,"","training"]],"serotiny.networks.layers":[[18,0,0,"-","convolution_block"],[18,0,0,"-","skip_connection"],[18,0,0,"-","spatial_pyramid_pool"]],"serotiny.networks.layers.convolution_block":[[18,1,1,"","ConvBlock"],[18,3,1,"","conv_block"]],"serotiny.networks.layers.convolution_block.ConvBlock":[[18,2,1,"","forward"],[18,4,1,"","training"]],"serotiny.networks.layers.skip_connection":[[18,1,1,"","SkipConnection"]],"serotiny.networks.layers.skip_connection.SkipConnection":[[18,2,1,"","forward"],[18,4,1,"","training"]],"serotiny.networks.layers.spatial_pyramid_pool":[[18,1,1,"","SpatialPyramidPool"],[18,3,1,"","spatial_pyramid_pool"]],"serotiny.networks.layers.spatial_pyramid_pool.SpatialPyramidPool":[[18,2,1,"","forward"],[18,4,1,"","training"]],"serotiny.networks.mlp":[[19,0,0,"-","mlp"]],"serotiny.networks.mlp.mlp":[[19,1,1,"","MLP"]],"serotiny.networks.mlp.mlp.MLP":[[19,2,1,"","forward"],[19,4,1,"","training"]],"serotiny.networks.utils":[[20,0,0,"-","weight_init"]],"serotiny.networks.utils.weight_init":[[20,3,1,"","weight_init"]],"serotiny.transforms":[[22,0,0,"-","dataframe"],[23,0,0,"-","image"]],"serotiny.transforms.dataframe":[[22,0,0,"-","transforms"]],"serotiny.transforms.dataframe.transforms":[[22,3,1,"","append_class_weights"],[22,3,1,"","append_labels_to_integers"],[22,3,1,"","append_one_hot"],[22,3,1,"","filter_columns"],[22,3,1,"","filter_rows"],[22,3,1,"","make_random_df"],[22,3,1,"","sample_n_each"],[22,3,1,"","split_dataframe"]],"serotiny.transforms.image":[[23,0,0,"-","crop"],[23,0,0,"-","normalize"],[23,0,0,"-","pad"],[23,0,0,"-","project"],[23,0,0,"-","swap"]],"serotiny.transforms.image.crop":[[23,1,1,"","CropCenter"]],"serotiny.transforms.image.normalize":[[23,1,1,"","NormalizeAbsolute"],[23,1,1,"","NormalizeMean"],[23,1,1,"","NormalizeMinMax"]],"serotiny.transforms.image.pad":[[23,1,1,"","ExpandColumns"],[23,1,1,"","ExpandTo"],[23,1,1,"","PadTo"],[23,3,1,"","expand_columns"],[23,3,1,"","expand_to"],[23,3,1,"","pull_to"],[23,3,1,"","split_number"],[23,3,1,"","to_tensor"]],"serotiny.transforms.image.project":[[23,1,1,"","Project"]],"serotiny.transforms.image.swap":[[23,1,1,"","Permute"],[23,1,1,"","SwapAxes"]],serotiny:[[7,0,0,"-","datamodules"],[8,0,0,"-","io"],[11,0,0,"-","losses"],[13,0,0,"-","models"],[16,0,0,"-","networks"],[21,0,0,"-","transforms"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[1,2,11,15,18,23],"06845":11,"1":[1,2,11,15,22,23],"10":2,"100":[2,22],"11":0,"1234":2,"13":0,"1337":2,"16":2,"1907":11,"1e":2,"2":[1,2,23],"256":19,"2d":18,"3":[1,2,17,18],"32":2,"3d":[15,17,18],"4":1,"42":22,"5":2,"50":[0,1],"64":2,"7":1,"abstract":10,"boolean":15,"case":[7,22],"class":[2,7,8,9,11,13,15,17,18,19,22,23],"default":[0,2,18,22],"do":[0,2],"float":[10,15,22],"function":[0,2,9,15,18,22],"import":2,"int":[8,10,15,17,18,22,23],"public":4,"return":[8,9,10,22],"true":[2,15,17,18,22,23],A:[2,7,9,10,22],AND:9,And:17,As:2,At:2,But:10,By:22,For:[0,1,2,9,10],If:[2,4,7,8,9,10,22],In:[0,9,22],It:[1,3,7,11,22],On:2,One:2,Or:4,That:[0,1],The:[0,4,8,9,10,22],There:2,These:[0,2],To:[1,4],With:2,_loss:[11,15],_partial_:2,_target_:2,a_datamodule_config:0,a_model_config:0,a_tiny_project:2,ab:11,abl:2,about:0,abov:[0,2],abstract_load:[8,9],ad:2,adam:[2,13,15],add:[0,2,22],addit:[0,7],addition:2,ahead:2,aic:9,aics_imag:8,aicsimag:8,aicsimageio:[8,10],all:[1,9,22],allencellmodel:[2,4],allow:1,allow_zero_length_dataloader_with_multiple_devic:15,also:[0,1,9,11],altern:2,alwai:[4,10],among:7,an:[0,2,8,11],ani:[0,7],anisotrop:15,append:[0,22],append_class_weight:22,append_labels_to_integ:22,append_one_hot:22,appli:[8,9,10,17,18,20],applic:2,appropri:[2,9],ar:[0,2,7,9,10,22],architectur:2,aren:[0,9],arg:7,argument:[0,1,7,9,22],arrai:[8,10,15,23],artifact:[2,3],arxiv:11,asid:7,assert:9,assign:8,assum:2,automat:2,avail:2,ax:23,axi:23,b:[1,2,22],backend:2,balanc:22,base:[7,9,10,11,13,15,17,18,19,22,23],base_model:[5,6,15],base_va:[6,13],basemodel:[13,15],baseva:15,bash:0,basic:15,basic_cnn:[6,16],basiccnn:17,batch:[11,13,15],batch_idx:13,batch_norm:18,batch_siz:[2,7],becaus:0,becom:2,befor:8,behavior:0,behind:0,belong:22,bernoulli:11,beta:15,better:2,between:[11,22],binari:10,block:[2,18],bool:[8,10,13,15,17,18,19,22],c:[1,22],cache_output:[13,15],calculate_elbo:15,call:[0,2,22],callabl:[8,10],can:[0,1,2,4,7,15,18,22],cblogloss:11,cell:[0,9],cellid:1,center_of_mass:23,chang:0,channel:[8,10,17,18],channel_nam:8,check:2,cli:[1,2],clip_max:23,clip_min:23,clip_quantil:23,clone:4,cnn:17,column:[1,7,8,9,22,23],columns_to_filt:9,com:[2,4],combin:9,come:0,command:[0,2,4],common:1,commun:2,complet:0,comput:[2,11],compute_loss:15,condit:9,config:0,configur:[0,3],configure_optim:13,consist:[9,10],constant:23,consum:2,contain:[2,7,8,9,10,22],content:5,context:0,continu:11,continuous_bernoulli:[5,6],contribut:2,conv_block:18,conv_forward:17,convblock:18,convert:22,convolut:18,convolution_block:[6,16],cookiecutt:2,copi:4,core:[7,9,13,22],correct:8,correspond:[2,8,9,22],could:2,coupl:2,covari:15,creat:[0,1,2,7,22],crop:[6,21],crop_raw:1,crop_seg:1,cropcent:23,cropi:23,cropx:23,cropz:23,csv:[1,2,9],curl:4,current:15,custom:2,d:22,dashboard:2,data:[0,7,9,10,18,22],datafram:[0,2,6,8,21],dataframe_dataset:[6,8],dataframedataset:[9,10],dataload:7,dataloader_kwarg:7,datamodul:[2,5,6],dataset:[2,7,9,22],decod:15,decoder_non_linear:15,deep:3,defin:1,depend:2,depth:17,describ:2,desir:9,destin:0,detail:22,determin:[8,10,15],diagon:[11,15],diagonal_gaussian_kl:11,dict:[7,9],differ:[0,2,9,22],dim:[10,19],dim_ord:8,dimens:[8,10,17,18,23],dimension:[17,18],directli:2,disk:8,diverg:11,doc:[2,7],doesn:8,don:4,done:[0,1],download:4,drive:3,dtype:[8,10],dure:2,e:[0,2,9,15],each:[0,2,7,8,9,17,22],earli:2,earlystop:2,easier:2,either:[2,4,7,9,15,22],enabl:22,enable_checkpoint:2,encod:[15,22],end:[1,9,10,22],endswith:[9,10,22],equal:22,etc:[2,7],eval:[0,2],evalu:0,everyth:7,exampl:[0,2],exclud:[1,9,10,22],execut:3,exist:22,expand_column:23,expand_to:23,expandcolumn:23,expanded_column:23,expandto:23,expect:9,experi:[1,2,3],experiment_nam:2,explicit:[10,22],expos:1,express:[9,10,22],extend:22,extract:10,fals:[1,8,10,15,17,18,22,23],familiar:2,fashion:2,feature_:2,file:[0,1,2,7,9,10],file_typ:10,filter:[1,9,10,22],filter_column:[1,9,22],filter_row:[1,22],find_lr_schedul:14,find_optim:14,first:[1,11,23],fix:15,flag:[0,8,10,15],flat_output:17,float32:[8,10],fold:7,folder:2,follow:2,forc:22,force_s:23,form:2,format:10,forward:[11,13,15,17,18,19],found:9,fraction:22,frame:[9,22],framework:[2,3],from:[0,2,7,8,9,10,14,15],futur:10,g:[0,2],gaussian:11,gener:[1,2,22],get:[0,14],git:4,github:[2,4],give:2,given:[0,7,8,9,14,22],govern:2,group:0,guid:4,ha:[0,2,22],half:22,have:[0,2,4,8,22],heavili:2,help:0,helper:1,here:[0,1,2,11],hidden:17,hidden_channel:[15,17],hidden_lay:[2,15,19],hood:0,hot:[10,22],how:[0,2],html:7,http:[2,4,7,11],hydra:[0,2,3],i:[9,15],id:2,id_label:[13,15],identifi:[2,22],ignor:[9,10,22],imag:[0,5,6,9,21],image_load:8,image_va:[6,13],imageva:15,img:8,implement:[1,2,10],implicitli:17,importantli:1,in_c:18,in_channel:[15,17],includ:[9,10,22],include_column:9,incomplet:2,independ:2,index:8,infer:[0,8],infer_dim:8,info:[0,2,7],inform:0,initi:[0,15],inner:2,input:[1,11,17,18,22],input_dim:[15,17],instal:0,instanc:10,instanti:[2,7,15,17,18],instead:2,integ:[8,22],intend:[0,2],interact:[0,2],interfac:15,interpol:2,intimid:0,involv:1,io:[2,5,6],ipynb:0,isotrop:[11,15],isotropic_gaussian_kl:11,its:[0,2,10,22],jupyt:2,just:[0,2],keep:22,kei:9,kept:10,kernel:18,kernel_s:[17,18],keyword:7,kl_diverg:[5,6],kld:15,know:1,kullback:11,kwarg:[7,13,15],label:[8,15],latenc:9,latent_dim:[0,2,15],launcher:0,layer:[6,16,17],learn:[0,3,15],learn_prior_logvar:15,least:2,leibler:11,less:22,let:1,level:2,leverag:[0,2,3],lightiningdatamodul:2,lightn:[2,3,7,13],lightningdatamodul:7,lightningmodul:[2,13],like:[0,2,10],line:[0,2],list:[2,7,8,9,10,22],ll:[0,2],load:[7,8,9,10],load_model_from_checkpoint:2,loadclass:10,loadcolumn:[2,10],loader:[2,7,8,9],loadimag:10,loadrandomtensor:10,log:[11,15],log_var:11,logic:9,logvar1:11,logvar2:11,logvar:[11,15],loop:2,loss:[5,6,15],loss_mask_label:15,lr:2,lr_schedul:14,m:[0,20],made:7,mai:2,main:[2,4],make:[0,2],make_dataload:7,make_notebook:0,make_random_df:[1,22],manag:3,mani:0,manifest:[1,2,7,9,10],manifest_datamodul:[5,6],manifestdatamodul:[2,7],map:8,mask:15,match:[9,10,22],matrix:15,max:[17,23],max_epoch:[0,2],max_pool_lay:[15,17],mean:[0,11,15],mechan:10,merg:1,method:[4,9],metric:2,might:[2,10],mimic:0,min_delta:2,ml:2,ml_op:[2,5,6],mlflow:[0,3],mlflow_util:[2,5,6],mlp:[6,16],mode:[11,15,17,18,23],model:[0,5,6,20],modifi:22,modul:[0,1,5],modular:2,modularli:2,monitor:2,more:[0,2,7,22],more_param:2,most:4,mostli:1,mseloss:15,mu1:11,mu2:11,mu:[11,15],multi:0,multipl:0,must:[8,9,10,22],my:2,my_input:1,my_output:1,n:23,n_row:[1,22],name:[2,7,8,9,10,14],necessarili:11,need:1,net:3,network:[5,6,15],neural:3,newli:2,nn:[11,15,17,18,19],non_linear:[15,17,18],none:[7,8,9,10,15,17,18,19,22,23],nonetyp:13,normal:[6,9,21],normalizeabsolut:23,normalizemean:23,normalizeminmax:23,note:[0,1,2],notebook:[0,2],now:[0,2,10],np:[8,10],num_class:10,num_work:[2,7],number:[8,10,17,18,22],numpi:[8,10,15],object:[2,10,23],obtain:0,ol:4,ome_tiff_read:10,omegaconf:2,ometiffread:10,omit:[0,2],onc:[0,4],one:[0,2,7,10,22],one_hot_encod:22,onli:[0,2,7,8,9,10,15,22],oper:[2,17,22],optim:[2,13,14,15],optimizer_nam:14,optimizer_util:[6,13],option:[0,7,8,9,10,15,17,18,22],order:[8,23],org:[7,11],organ:2,other:[2,10],otherwis:9,our:2,out:2,out_c:18,out_pool_s:18,output:[8,13,17,18],output_dim:17,output_dtyp:8,output_path:1,overrid:[0,2],own:2,p:2,packag:5,package_nam:2,pad:[6,18,21],padto:23,panda:[9,10,22],param1:2,param2:2,paramet:[0,2,7,8,9,10,11,15,17,18,22],parquet:[7,9],parse_batch:[13,15],part:2,particular:0,partit:1,pass:7,path:[0,2,7,8,9,10],pathlib:[8,9],patienc:2,pattern:9,pd:[9,22],per:22,perform:22,permut:23,pip:4,pipelin:22,place:2,placehold:1,point:[1,9],pool:17,popul:0,port:2,possibl:[2,10],practic:9,preced:[1,9],precis:15,predict:[2,3],predict_dataload:7,prefer:4,prepare_data_per_nod:15,previou:1,prior:15,prior_logvar:15,prior_mod:15,prob:11,process:[2,4],project:[0,3,6,21],project_descript:2,project_nam:2,propos:[2,11],provid:[1,2,10,22],pull_to:23,purpos:2,py:[2,4],pyramid_pool_split:17,python:[2,4],pytorch:[2,3,7,9,15],pytorch_lightn:[2,7,13],queri:9,random:[1,22],random_imag:[8,9],re:[0,1],read:[2,9],read_csv:9,read_datafram:9,read_parquet:9,reader:[6,8,10],readi:2,reason:0,recent:4,recogn:2,recommend:2,reconstruct:15,reconstruction_loss:15,reconstruction_reduc:15,reduct:11,regex:[9,10,22],regular:[9,10,22],reli:2,remain:[9,10,22],replac:22,repo:4,repositori:4,requir:9,required_column:9,respect:[0,9,15],result:[1,2,9,10,22],retriev:[8,10,15],return_channel:8,return_s:17,return_split:[1,22],return_torch:23,right:2,row:[1,10,22,23],run:[0,2,4],run_nam:2,s:[0,2,7,22],same:15,sampl:22,sample_n_each:22,sample_z:15,save:8,sc:0,scale:23,schedul:14,scheduler_nam:14,script:0,second:[1,11,23],section:0,see:[0,2,7,22],seed:[0,2,22],select:[2,22],select_channel:[8,10],separ:22,sequenc:[7,8,9,10,13,15,17,18,22],serotini:[1,2,4],serotiny_bash:0,server:2,session:0,set:[2,3],setup:[2,4],shall:10,shell:0,shot:2,should:2,shuffl:7,signifi:1,simpl:2,singl:7,skip_connect:[6,15,16,17],skipconnect:18,slower:9,smaller:22,so:[0,2],sourc:[7,8,9,10,11,13,14,15,17,18,19,20,22,23],spatial_pyramid_pool:[6,16],spatialpyramidpool:18,specif:[2,7,10],specifi:[0,2],split:[1,2,7,22],split_column:[2,7],split_datafram:[1,22],split_numb:23,stabl:7,stand:2,standard:2,start:[0,1,9,10,22],startswith:[2,9,10,22],step:1,stop:2,store:[2,9],str:[7,8,9,10,11,13,15,17,18,22],straightforward:0,stratifi:22,string:[9,10,15,22],structur:3,subclass:[10,15],submodul:[5,6,16,21],subpackag:5,substr:[9,10,22],sum:[11,15],suppli:[9,10,22],support:[9,10,15],swap:[2,6,21],swapax:23,sweep:0,syntax:[0,2],t:[0,4,8,9],tab:0,tabular_va:[6,13],tabularva:[2,15],take:9,tarbal:4,target:11,target_dim:23,templat:2,tensor:[8,10,11,15],term:15,termin:4,test:[2,3,7,13,15,22],test_dataload:7,test_epoch_end:13,test_step:13,than:22,the_run_id:2,the_tracking_uri:2,thei:2,them:[0,1,2,10],thi:[0,1,2,4,8,9,10,22],third:1,three:[2,7,22],through:4,thrown:9,thu:2,tiff:[8,10],tiff_writ:8,tini:2,tinker:0,to_tensor:23,todo:22,toggl:22,too:10,tool:[1,3],top:2,torch:[2,7,8,9,11,13,14,15,17,18,19],track:3,tracking_uri:2,train:[1,3,7,13,15,17,18,19,22],train_dataload:7,train_epoch_end:13,train_frac:[1,22],trainer:0,training_step:13,transform:[5,6,8,10],trivial:2,two:11,type:[8,9,10,13,15],under:[0,2],understand:[0,2],union:[7,8,9,10,15,17,22],uniqu:22,up_conv:[17,18],upon:10,upsampl:22,upsample_lay:17,uri:2,url:2,us:[0,1,2,7,8,9,10,15,22],usag:[2,20],use_amp:15,user:1,util:[0,5,6,7,9,13,16,18],vae:[6,13],val:[1,7,22],val_dataload:7,val_frac:22,val_loss:2,valid:22,validation_epoch_end:13,validation_step:13,valu:[0,2,10,15,18,22,23],valueerror:9,varianc:[11,15],ve:0,vector:10,veri:2,wai:2,want:[0,1,2],we:[1,2,9],weight:[15,22],weight_init:[6,16],well:[0,2],what:[0,2],when:[0,2,7,8],where:[2,8,22],whether:[8,10,15,22],which:[0,1,2,7,9,10,15,17,22],wise:11,within:2,work:[2,11],would:2,wrangl:[0,22],wrap:9,wrapped_modul:18,write:[2,8],x1:19,x2:19,x:[2,13,15,17,18],x_dim:[2,15],x_hat:15,x_label:[2,13,15],y_encoded_label:10,yaml:2,you:[0,1,2,4],your:[0,2,4],your_data_config_nam:2,your_model_config_nam:2,yourcustomcallback:2,zarr:10},titles:["serotiny CLI","Dataframe wrangling","Getting started","serotiny","Installation","serotiny","serotiny package","serotiny.datamodules package","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.loaders package","serotiny.losses package","serotiny.ml_ops package","serotiny.models package","serotiny.models.utils package","serotiny.models.vae package","serotiny.networks package","serotiny.networks.basic_cnn package","serotiny.networks.layers package","serotiny.networks.mlp package","serotiny.networks.utils package","serotiny.transforms package","serotiny.transforms.dataframe package","serotiny.transforms.image package"],titleterms:{"class":10,The:2,abstract_load:10,base_model:13,base_va:15,basic_cnn:17,call:1,callback:2,chain:1,cli:0,column:10,config:2,configur:2,content:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],continuous_bernoulli:11,convolution_block:18,crop:23,data:2,datafram:[1,9,10,22],dataframe_dataset:9,datamodul:7,debug:0,from:4,get:2,group:2,imag:[8,10,23],image_va:15,instal:4,io:[8,9,10],kl_diverg:11,layer:18,load:2,loader:10,loss:11,manifest_datamodul:7,ml:0,ml_op:12,mlflow:2,mlflow_util:12,mlp:19,model:[2,13,14,15],modul:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],multipl:1,network:[16,17,18,19,20],normal:23,oper:0,optimizer_util:14,packag:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],pad:23,pipelin:1,predict:0,project:[2,23],random_imag:10,reader:9,releas:4,serotini:[0,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],singl:1,skip_connect:18,sourc:4,spatial_pyramid_pool:18,stabl:4,start:2,structur:2,submodul:[7,8,9,10,11,12,13,14,15,17,18,19,20,22,23],subpackag:[6,8,9,13,16,21],swap:23,tabular_va:15,test:[0,1],train:[0,2],trainer:2,transform:[1,21,22,23],util:[12,14,20],vae:15,weight_init:20,wrangl:1}})