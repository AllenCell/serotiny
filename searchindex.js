Search.setIndex({docnames:["cli","contributing","dataframe_transforms","dynamic_imports","example_workflows","feature_extraction","image_transforms","index","installation","model_zoo","modules","quickstart","serotiny","serotiny.datamodules","serotiny.imports","serotiny.io","serotiny.io.dataframe","serotiny.io.dataframe.loaders","serotiny.losses","serotiny.metrics","serotiny.models","serotiny.models.callbacks","serotiny.models.inference","serotiny.models.unet","serotiny.models.utils","serotiny.models.vae","serotiny.networks","serotiny.networks.basic_cnn","serotiny.networks.layers","serotiny.networks.mlp","serotiny.networks.unet","serotiny.networks.utils","serotiny.transforms","serotiny.transforms.dataframe","serotiny.transforms.image","serotiny.transforms.pipeline"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","contributing.rst","dataframe_transforms.rst","dynamic_imports.rst","example_workflows.rst","feature_extraction.rst","image_transforms.rst","index.rst","installation.rst","model_zoo.rst","modules.rst","quickstart.rst","serotiny.rst","serotiny.datamodules.rst","serotiny.imports.rst","serotiny.io.rst","serotiny.io.dataframe.rst","serotiny.io.dataframe.loaders.rst","serotiny.losses.rst","serotiny.metrics.rst","serotiny.models.rst","serotiny.models.callbacks.rst","serotiny.models.inference.rst","serotiny.models.unet.rst","serotiny.models.utils.rst","serotiny.models.vae.rst","serotiny.networks.rst","serotiny.networks.basic_cnn.rst","serotiny.networks.layers.rst","serotiny.networks.mlp.rst","serotiny.networks.unet.rst","serotiny.networks.utils.rst","serotiny.transforms.rst","serotiny.transforms.dataframe.rst","serotiny.transforms.image.rst","serotiny.transforms.pipeline.rst"],objects:{"":[[12,0,0,"-","serotiny"]],"serotiny.imports":[[14,0,0,"-","dynamic_imports"]],"serotiny.imports.dynamic_imports":[[14,1,1,"","bind"],[14,1,1,"","get_load_method_and_args"],[14,1,1,"","get_name_and_arguments"],[14,1,1,"","get_name_from_path"],[14,1,1,"","init"],[14,1,1,"","invoke"],[14,1,1,"","load_config"],[14,1,1,"","load_multiple"],[14,1,1,"","module_get"],[14,1,1,"","module_or_path"],[14,1,1,"","module_path"],[14,1,1,"","search_modules"]],"serotiny.io":[[15,0,0,"-","buffered_patch_dataset"],[15,0,0,"-","image"],[15,0,0,"-","quilt"]],"serotiny.io.buffered_patch_dataset":[[15,2,1,"","BufferedPatchDataset"]],"serotiny.io.buffered_patch_dataset.BufferedPatchDataset":[[15,3,1,"","get_buffer_history"],[15,3,1,"","get_patch"],[15,3,1,"","get_random_patch"],[15,3,1,"","insert_new_element_into_buffer"]],"serotiny.io.dataframe":[[16,0,0,"-","dataframe_dataset"]],"serotiny.io.dataframe.dataframe_dataset":[[16,2,1,"","DataframeDataset"]],"serotiny.io.image":[[15,1,1,"","image_loader"],[15,1,1,"","infer_dims"],[15,1,1,"","tiff_writer"]],"serotiny.io.quilt":[[15,1,1,"","download_quilt_data"]],"serotiny.losses":[[18,0,0,"-","continuous_bernoulli"],[18,0,0,"-","kl_divergence"]],"serotiny.losses.continuous_bernoulli":[[18,2,1,"","CBLogLoss"]],"serotiny.losses.continuous_bernoulli.CBLogLoss":[[18,3,1,"","forward"],[18,4,1,"","reduction"]],"serotiny.losses.kl_divergence":[[18,1,1,"","diagonal_gaussian_kl"],[18,1,1,"","isotropic_gaussian_kl"]],"serotiny.metrics":[[19,2,1,"","InceptionV3"],[19,1,1,"","calculate_blur"],[19,0,0,"-","calculate_blur"],[19,1,1,"","calculate_fid"],[19,0,0,"-","calculate_fid"],[19,0,0,"-","inception"],[19,0,0,"-","pearson"]],"serotiny.metrics.InceptionV3":[[19,4,1,"","BLOCK_INDEX_BY_DIM"],[19,4,1,"","DEFAULT_BLOCK_INDEX"],[19,3,1,"","forward"],[19,4,1,"","training"]],"serotiny.metrics.calculate_blur":[[19,1,1,"","calculate_blur"]],"serotiny.metrics.calculate_fid":[[19,1,1,"","calculate_fid"],[19,1,1,"","get_activations"]],"serotiny.metrics.inception":[[19,2,1,"","InceptionV3"]],"serotiny.metrics.inception.InceptionV3":[[19,4,1,"","BLOCK_INDEX_BY_DIM"],[19,4,1,"","DEFAULT_BLOCK_INDEX"],[19,3,1,"","forward"],[19,4,1,"","training"]],"serotiny.metrics.pearson":[[19,1,1,"","pearson_correlation"]],"serotiny.models":[[24,0,0,"-","utils"]],"serotiny.models.utils":[[24,0,0,"-","optimizer_utils"]],"serotiny.models.utils.optimizer_utils":[[24,1,1,"","find_lr_scheduler"],[24,1,1,"","find_optimizer"]],"serotiny.networks":[[29,0,0,"-","mlp"],[30,0,0,"-","unet"],[31,0,0,"-","utils"]],"serotiny.networks.mlp":[[29,0,0,"-","mlp"]],"serotiny.networks.mlp.mlp":[[29,2,1,"","MLP"]],"serotiny.networks.mlp.mlp.MLP":[[29,3,1,"","forward"],[29,4,1,"","training"]],"serotiny.networks.unet":[[30,0,0,"-","double_convolution"],[30,0,0,"-","unet_3d"],[30,0,0,"-","unet_downconv"],[30,0,0,"-","unet_upconv"]],"serotiny.networks.unet.double_convolution":[[30,2,1,"","DoubleConvolution"]],"serotiny.networks.unet.double_convolution.DoubleConvolution":[[30,3,1,"","forward"],[30,4,1,"","training"]],"serotiny.networks.unet.unet_3d":[[30,2,1,"","Unet3d"]],"serotiny.networks.unet.unet_3d.Unet3d":[[30,3,1,"","forward"],[30,4,1,"","num_output_channels"],[30,3,1,"","print_network"],[30,4,1,"","training"]],"serotiny.networks.unet.unet_downconv":[[30,2,1,"","DownConvolution"]],"serotiny.networks.unet.unet_downconv.DownConvolution":[[30,3,1,"","forward"],[30,4,1,"","training"]],"serotiny.networks.unet.unet_upconv":[[30,2,1,"","UpConvolution"]],"serotiny.networks.unet.unet_upconv.UpConvolution":[[30,3,1,"","forward"],[30,4,1,"","training"]],"serotiny.networks.utils":[[31,0,0,"-","weight_init"]],"serotiny.networks.utils.weight_init":[[31,1,1,"","weight_init"]],"serotiny.transforms":[[33,0,0,"-","dataframe"],[34,0,0,"-","image"]],"serotiny.transforms.dataframe":[[33,0,0,"-","transforms"]],"serotiny.transforms.dataframe.transforms":[[33,1,1,"","append_class_weights"],[33,1,1,"","append_labels_to_integers"],[33,1,1,"","append_one_hot"],[33,1,1,"","filter_columns"],[33,1,1,"","filter_rows"],[33,1,1,"","make_random_df"],[33,1,1,"","sample_n_each"],[33,1,1,"","split_dataframe"]],"serotiny.transforms.image":[[34,0,0,"-","align"],[34,0,0,"-","feature_extraction"],[34,0,0,"-","normalize"],[34,0,0,"-","pad"],[34,0,0,"-","project"],[34,0,0,"-","resize"],[34,0,0,"-","swap"]],"serotiny.transforms.image.align":[[34,1,1,"","align_image_2d"]],"serotiny.transforms.image.feature_extraction":[[34,1,1,"","angle"],[34,1,1,"","bbox"],[34,1,1,"","center_of_mass"],[34,1,1,"","dillated_bbox"],[34,1,1,"","min_max"],[34,1,1,"","percentile"],[34,1,1,"","shcoeffs"]],"serotiny.transforms.image.normalize":[[34,2,1,"","NormalizeAbsolute"],[34,2,1,"","NormalizeMean"],[34,2,1,"","NormalizeMinMax"]],"serotiny.transforms.image.pad":[[34,2,1,"","ExpandColumns"],[34,2,1,"","ExpandTo"],[34,2,1,"","PadTo"],[34,1,1,"","expand_columns"],[34,1,1,"","expand_to"],[34,1,1,"","pull_to"],[34,1,1,"","split_number"],[34,1,1,"","to_tensor"]],"serotiny.transforms.image.project":[[34,2,1,"","Project"]],"serotiny.transforms.image.resize":[[34,2,1,"","CropCenter"],[34,2,1,"","ResizeBy"],[34,2,1,"","ResizeTo"]],"serotiny.transforms.image.swap":[[34,2,1,"","Permute"],[34,2,1,"","SwapAxes"]],serotiny:[[12,1,1,"","get_module_version"],[14,0,0,"-","imports"],[15,0,0,"-","io"],[18,0,0,"-","losses"],[19,0,0,"-","metrics"],[32,0,0,"-","transforms"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"0":[2,3,6,18,19,30,34],"00638175":3,"0151588":3,"01525316":3,"04597":30,"05":6,"05956044":3,"06":19,"06845":18,"1":[2,3,5,6,15,18,19,30,33,34],"10":[0,3,6],"100":[5,6,33],"12":30,"14726909":3,"1505":30,"19":3,"1907":18,"192":19,"1e":19,"2":[2,3,6,19,30,34],"20":3,"2048":19,"20821983":3,"24":30,"256":29,"29300648":3,"299":19,"3":[2,3,6,19,30],"32":15,"32604151":3,"32854592":3,"3287971":3,"344":19,"37383497":3,"4":[0,2,3,6,30,34],"41411613":3,"42":33,"48":30,"48772363":3,"49404687":3,"5":[3,6,34],"50":[2,6],"54736776":3,"56047808":3,"59323792":3,"59621":3,"6":30,"60767339":3,"64":[3,15,19,30],"66560783":3,"68928192":3,"7":[2,3],"768":19,"78241693":3,"78599943":3,"8":[3,5],"82680974":3,"8956135":3,"89774285":3,"95":6,"96":30,"98165106":3,"98437869":3,"98945791":3,"98970493":3,"break":[0,5,6],"case":[3,6,33],"class":[3,9,14,15,16,18,19,29,30,33,34],"default":[9,33],"do":0,"final":[0,6,19,30],"float":[19,33,34],"function":[0,3,5,6,9,14,16,19,33],"import":[5,6,10,11,12],"int":[15,19,30,33,34],"new":[1,15],"public":8,"return":[3,5,6,12,15,19,33],"short":1,"true":[5,6,14,15,19,30,33,34],"try":[5,6],A:[1,6,16,19,33,34],As:[5,6,19],By:[9,33],For:[0,2,3,5,6],If:[0,3,8,9,15,19,33],In:[6,16,30,33],It:[1,2,6,18,33],No:30,Or:8,That:2,The:[0,3,5,6,8,15,16,19,33,34],Then:1,There:3,To:[2,3,6,8],_3d:3,_get_checkpoint:11,_loss:18,ab:18,abl:19,abov:[0,6,9],abstract_load:[15,16],accept:0,activ:19,ad:0,adapt:3,add:[0,1,30,33],addition:[3,6],adjust:34,adjust_for_skew:34,after:[0,9],again:6,agglomer:12,ahead:[3,5,6],aic:[12,15,16,34],aics_imag:15,aicsimag:15,aicsimageio:[5,6,15],aicsshparam:34,align:[12,32],align_image_2d:34,alignment_2d:34,all:[0,1,2,3,6,11,19,33],allencel:[15,34],allencellmodel:8,allow:[2,3,6],almost:3,also:[0,1,2,3,6,9,18],alwai:[1,8],an:[0,3,5,6,9,12,15,18,19,30],anaconda:1,angl:34,ani:[0,6],api:5,append:[0,33],append_class_weight:33,append_labels_to_integ:33,append_one_hot:33,appli:[0,15,31],appreci:1,appropri:16,ar:[1,3,6,9,19,30,33],arbitrari:19,architectur:30,argument:[0,2,3,6,33],arrai:[3,15,19,34],arraylik:[15,34],arxiv:[18,30],ascend:19,assign:15,assum:11,attrib:15,attribut:[3,15],auto:30,autograd:19,aux:19,averag:[19,30],avoid:[5,6],ax:34,axi:[6,34],b:[1,2,19,33],backend:[5,6],balanc:33,base:[3,15,16,18,19,29,30,33,34],base_va:[12,20],basic_cnn:[12,26],batch:[18,19],batch_siz:19,bbox:[5,34],becaus:6,befor:[15,19],behaviour:6,being:6,belong:33,bernoulli:18,between:[15,18,19,30,33],bilinearli:19,bind:[5,6,14],bit:1,blob:[19,34],block:19,block_index_by_dim:19,blurri:19,bn:30,bool:[15,19,29,30,33,34],both:6,bound:[5,34],bounding_box:5,box:[5,34],br:1,branch:[1,30],bucket:15,buffer:15,buffer_index:15,buffer_s:15,buffer_switch_interv:15,buffered_patch_dataset:[10,12],bufferedpatchdataset:15,bugfix:1,build:[1,19],bump2vers:1,bx3xhxw:19,c:[2,33],c_1:19,c_2:19,cach:9,calcul:[19,30],calculate_blur:[10,12],calculate_fid:[10,12],call:[3,5,6,33],callabl:[5,15],callback:[9,12,20],can:[0,1,2,3,5,6,8,9,33],cblogloss:18,cd:1,cell:[12,16],cell_id:6,cellid:[2,5],center:34,center_of_mass:[5,6,34],centric:[3,6],certain:5,chang:1,channel:[3,5,6,15,34],channel_fan:[3,30],channel_fan_top:[3,30],channel_map:34,channel_nam:15,check:1,checkout:1,checkpoint:9,chunk:0,classifi:19,classnam:14,cli:[2,5,6,9],clip_max:[6,34],clip_min:[6,34],clip_quantil:[6,34],clone:[1,8],coeffici:[5,19,34],col:6,colon:0,column:[2,5,6,15,16,33,34],com:[1,8,19,34],come:0,command:[0,6,8],commit:1,common:[2,6],complet:12,complex:[0,6],comput:[5,18,34],compute_lcc:34,concat:6,concaten:[6,30],conditional_va:[12,20],condprior_va:[12,20],config:[3,5,6,9,11,14],configur:[0,3],conform:5,connect:19,consist:16,constant:34,contain:[0,3,5,6,11,15,16,19,33,34],content:10,continu:[5,6,18],continuous_bernoulli:[10,12],conv_out:30,conveni:6,convert:33,convolut:[19,30],convolution_block:[12,26],copi:8,core:33,correct:15,correl:19,correspod:34,correspond:[5,6,9,15,16,19,33],covari:19,creat:[1,2,33],credit:1,crop:[6,34],crop_and_norm:6,crop_raw:2,crop_seg:2,cropcent:[6,34],cropi:[6,34],cropx:[6,34],cropz:[6,34],cross:30,csv:[2,5,6],cuda:19,cumbersom:0,curl:8,current_depth:30,d:[19,33],dash:0,data:[5,6,12,15,16,19,33],data_save_loc:15,datafram:[0,6,12,15,32],dataframe_dataset:[12,15],dataframedataset:16,datamodul:[10,11,12],dataset:[5,6,15,16,33],debug:[5,6],default_block_index:19,defin:[0,2],depend:[6,19,30],depth:[3,30],describ:[3,6,9],descript:1,desir:[0,9],detail:[1,33],determin:[9,15],dev:1,develop:[1,3],diagon:18,diagonal_gaussian_kl:18,dict:[5,6,16,34],dictionari:[0,3,5,6,34],differ:[0,6,33],dilat:34,dillat:[6,34],dillated_bbox:[5,34],dim:[19,29],dim_ord:15,dimens:[15,19,34],dimension:19,direct:5,directli:6,directori:9,disabl:15,disk:[5,6,15],distanc:19,dive:6,diverg:18,dna_raw:6,dna_seg:6,doesn:15,don:[0,8,30],done:[1,2,30],dot:0,doubl:30,double_conv:30,double_convolut:[12,26],doubleconvolut:30,dougal:19,down:3,downconv:30,downconvolut:30,download:[8,15],download_quilt_data:15,downstream:5,dtype:[3,15],dummi:[10,12],dynam:[5,6],dynamic_import:[10,12],e:[1,5,6,14],each:[3,5,6,9,15,16,19,33],easi:3,edit:1,either:[5,6,8,33],element:[15,34],enabl:[5,6,33],encod:33,end:[0,2,33],endswith:33,enough:[0,3],enter:0,entir:30,environ:[1,9],ep:19,equal:33,equival:6,error:[5,6],etc:5,everi:[1,6],everyth:4,ex:1,exampl:[3,9,15,30],exchang:15,exclud:[2,33],exist:[5,33],expand_column:34,expand_to:34,expandcolumn:34,expanded_column:34,expandto:34,expect:[3,6,19],experi:2,explain:5,explicit:33,expos:2,express:33,extend:33,extract:[15,34],extract_featur:5,factor:34,fals:[2,5,6,14,15,19,33,34],featur:[1,19,34],feature_extract:[0,12,32],fed:19,feed:19,field:[0,6],file:[0,1,2,3,5,6,11,15,16],filenam:[0,6],filter:[2,6,33],filter_column:[2,33],filter_row:[2,33],find:5,find_lr_schedul:24,find_optim:24,finetun:19,first:[0,2,18,19,30,34],flag:15,flexibl:6,float32:15,folder:9,follow:30,forc:33,force_s:34,fork:1,forward:[18,19,29,30],found:1,fraction:33,frame:33,framework:3,frechet:19,from:[3,6,11,15,16,24,34],full_load:3,fulli:19,further:3,g:[5,6,14],gaussian:[10,12,18,19],gener:[2,5,6,19,33],get:[19,24],get_activ:19,get_buffer_histori:15,get_load_method_and_arg:14,get_model:[9,11],get_module_vers:12,get_name_and_argu:14,get_name_from_path:14,get_patch:15,get_predict:19,get_random_patch:15,get_trainer_at_checkpoint:11,gh:1,git:[1,8],github:[1,8,19,34],given:[1,6,14,15,19,24,33,34],gpu:19,gradient:19,greatli:1,guid:8,ha:[0,3,15,33],half:33,handl:[1,19],hardwar:19,harmon:[5,34],have:[0,8,9,15,33],height:19,help:[0,1],helper:2,here:[1,2,6,9,18],hi:19,hidden_lay:29,high:3,hipsc_single_cell_image_dataset:15,home:9,hot:33,how:[0,1,3,4],howev:[0,6],html:1,http:[8,18,19,30,34],i:[0,6],idea:3,identifi:[9,33],ignor:33,ignore_warn:15,imag:[0,10,12,16,19,32],image2d:[15,16],image3d:[15,16],image_load:15,image_read:[5,6],image_va:[12,20],img:[15,34],img_path:6,implement:[2,19,30],importantli:2,incept:[10,12],inceptionv3:19,includ:[1,5,6,33],include_col:[5,6],index:[6,7,15,19],index_col:[5,6],indic:[15,19,34],individual_arg:6,infer:[0,12,15,20],infer_dim:15,inform:0,init:[6,14],initi:[5,6],inp:19,input:[0,2,5,6,18,19,33],input_channel:0,input_manifest:[5,6],insert:15,insert_new_element_into_buff:15,insid:9,instal:1,instanc:19,instanti:[3,6],instead:[5,6],integ:[15,33],integr:3,intend:3,interest:[3,6],intermedi:[5,6],interpret:[0,3],invok:14,involv:2,io:[5,6,10,12],isotrop:18,isotropic_gaussian_kl:18,item:15,iter:[5,6],its:[3,6,9,33],j:19,just:[0,3,30],k:30,keep:[19,33],kei:[5,6,14,16,34],kernel_s:30,kernel_size_doubleconv:30,kernel_size_finalconv:30,kernel_size_pool:30,kernel_size_upconv:30,keyword:[3,6],kl_diverg:[10,12],know:[0,2,6],kullback:18,l0:30,l1:30,l2:30,l3:30,l4:30,l:3,label:15,last:[6,15],latent_dimens:0,layer:[12,19,26,30],least:[5,6],leibler:18,less:33,let:[2,3,6,15],level:30,leverag:3,lie:19,like:[3,9,19],line:[0,6,19],lint:1,list:[3,5,6,15,19,33,34],littl:1,lmax:34,load:[3,5,15,16],load_config:[3,14],load_multipl:14,loadabl:0,loaded_ok:14,loader:[15,16],local:1,log:18,log_var:18,logic:6,logvar1:18,logvar2:18,logvar:18,loop:19,loss:[10,12],low:3,lr_schedul:24,m2r:1,m:[0,1,31],main:[8,34],maintain:1,major:[1,34],make:[0,1,3,6],make_random_df:[2,33],make_uniqu:34,mani:3,manifest:[2,5,6,16],manifest_datamodul:[10,12],map:[15,19],mask:6,mass:34,master:19,match:33,matrix:19,matter:6,max:[19,30,34],maxim:3,maximum:34,mean:[18,19,30],membrane_raw:6,membrane_seg:[5,6],merg:[0,2,5,6],method:[8,16,34],metric:[10,12],might:[0,19],min:[19,34],min_max:34,minimum:34,minmaxnorm:6,minor:1,misc:0,mismatch:30,ml:12,mlp:[12,26],mode:[1,18,34],model:[0,10,12,19,31],model_class:9,model_id:11,model_path:11,model_zoo:[9,11],model_zoo_path:11,modifi:33,modul:[0,2,5,7,10],module_get:14,module_or_path:14,module_path:14,more:[0,15,30,33],moreov:3,most:8,mostli:2,mu1:[18,19],mu2:[18,19],mu:18,mu_1:19,mu_2:19,multipl:[0,6],multipli:6,multiprocess:[5,6],multivari:19,must:[3,6,15,19,33],my:9,my_computed_featur:5,my_input:2,my_manifest:[5,6],my_output:2,my_randint:3,my_randn:3,my_version_str:9,n:[19,34],n_imag:19,n_in:30,n_out:30,n_row:[2,33],n_worker:[5,6],name:[5,6,15,24,34],nd:15,ndarrai:[15,19],nearest:34,necessarili:18,need:[2,3,5,6,19],nest:0,net:[3,19],net_config:3,network:[3,10,12,19],next:30,nf:15,nn:[18,19,29,30],none:[3,15,29,33,34],normal:[6,12,16,19,32],normalize_input:19,normalizeabsolut:34,normalized_raw:6,normalizemean:34,normalizeminmax:34,notat:0,note:[0,2,3,6],now:[1,3,5,30],np:15,num:19,num_input_channel:[3,30],num_output_channel:[3,30],number:[3,5,6,15,19,33],numpi:[3,6,15,19,34],object:[15,34],obtain:[0,3,6],often:5,ol:8,ome_tiff_read:[5,6],ometiffread:[5,6],onc:[5,6,8,9],one:[0,5,6,11,30,33],one_hot_encod:33,ones:[3,5],ones_config:3,onli:[3,15,33],oper:[0,5,6,33],optim:24,optimizer_nam:24,optimizer_util:[12,20],option:[5,6,9,15,30,33,34],order:[6,15,34],org:[18,30],origin:[1,3,30],other:[1,6,30],otherwis:9,our:6,out_step:19,output:[5,15,19],output_block:19,output_dtyp:15,output_manifest:6,output_path:[2,5,6],over:19,overrid:0,p:30,packag:[1,10],pad:[6,12,30,32],padding_doubleconv:30,padding_finalconv:30,padding_pool:30,padding_upconv:30,padto:34,page:[6,7],pair:34,panda:[16,33],paper:30,parallel:[5,6],param:19,paramet:[0,3,6,9,15,16,18,19,30,33,34],part:6,partial:3,partit:[0,2],pass:[1,3,6,9],patch:[1,10,12,15],patch_column:15,patch_shap:15,path:[0,5,6,9,14,15,16],path_col:5,pathlib:15,pd:[16,33],pdf:30,pearson:[10,12],pearson_correl:19,per:[0,6,33],perc:34,percentil:34,perform:[30,33],permut:34,pip:[1,8],pipelin:[0,12,32,33],place:34,placehold:2,plot_xhat:[12,20],point:[2,5,6,16],pool:[19,30],pool_3:19,posit:3,possibl:[0,1,9,19],practic:16,precalcul:19,preced:2,predict:0,prefer:[0,8],prefix:34,prepend:34,pretrain:19,previou:[2,6,9],primit:12,print_network:30,prob:18,process:[5,6,8],produc:[5,6],project:[1,12,32],propos:18,provid:[2,3,4,5,6,15,33],publish:1,pull:1,pull_to:34,push:1,py:[8,19,30,34],pypi:1,python:[1,3,8],pytorch:16,quantil:5,queri:19,quilt:[10,12],quit:0,rais:[5,6],randint:3,randint_config:3,randn:3,randn_config:3,random:[2,3,15,33],random_imag:[15,16],rang:19,raw:1,re:[1,2,6],read:[1,3],reader:[5,6,12,15],readi:1,reason:[0,19],recent:8,recommend:1,recurr:14,reduct:18,refer:6,regex:33,regist:6,regular:33,releas:1,reli:6,remain:33,remind:1,replac:33,repo:[1,8],report:19,repositori:8,represent:19,request:1,requir:[6,19],requires_grad:19,resiz:[6,12,19,32],resize_input:19,resizebi:34,resizeto:34,resolv:1,result:[2,3,5,6,33,34],retriev:[5,6,14,15],return_channel:15,return_merg:[5,6],return_split:[2,33],return_torch:34,right:30,root:9,rotat:34,row:[2,33,34],run:[1,5,6,8],s3:15,s:[0,1,3,5,6,9,30,33],same:6,sampl:[15,19,33],sample_n_each:33,save:[5,6,15],scale:34,scenario:0,schedul:24,scheduler_nam:24,script:0,search:7,search_modul:14,second:[2,18,19,34],section:9,see:[30,33,34],seed:33,select:[5,6,19,33],select_channel:15,separ:[30,33],sequenc:[6,15,33,34],sequenti:[12,26],serotini:[1,2,3,5,6,8,9,11],serotiny_zoo_root:9,serv:3,set:[1,5,6,9,15,19],setup:8,sever:6,shape:[3,15,19,34],share:6,shcoeff:34,should:[3,19],show:4,shparam:34,shuffle_imag:15,sigma1:19,sigma2:19,sigma:34,signatur:3,signifi:2,similar:5,simpli:[3,6,12],singl:[0,6],situat:0,size:[3,15,19],skew:34,skew_adjust:34,skip_if_exist:[5,6],smaller:33,snakemak:3,so:[19,30],some:[0,5,14],someon:3,sort:19,sourc:[12,14,15,16,18,19,24,29,30,31,33,34],spatial_pyramid_pool:[12,26],special:[3,6],specifi:[0,3,5,9],spheric:[5,34],spirit:5,split:[2,19,33],split_datafram:[2,33],split_numb:34,sqrt:19,stabl:19,start:[2,5,6,33],startswith:33,statist:19,step:[2,5,6],still:0,store:[5,6,9,15],str:[14,15,18,30,33,34],straightforward:0,stratifi:33,strictli:19,stride:30,stride_doubleconv:30,stride_finalconv:30,stride_pool:30,stride_upconv:30,string:[0,9,14,33],structur:[5,6,34],structure_raw:6,structure_seg:6,structuring_el:[5,34],subfold:9,submit:1,submodul:[10,12,20,26,32],subpackag:10,subset:6,substr:33,sum:18,suppli:33,suppress:15,supress:15,sure:1,sutherland:19,swap:[12,32],swapax:34,t:[0,8,15,30],tabular_conditional_va:[12,20],tabular_condprior_va:[12,20],tabular_va:[12,20],tabularva:9,tag:1,take:[0,3,6,9],tarbal:8,target:18,target_dim:34,task:5,tensor:[15,18,19,34],termin:8,test:[1,19,33],than:[15,33],the_input_image_col:5,thei:1,them:2,thi:[0,1,2,3,5,6,8,9,12,15,16,30,33],third:2,those:6,three:33,through:[1,8],thu:3,ti:4,tiff:[1,15],tiff_writ:15,tile_predict:[12,20],to_tensor:34,todo:[4,30,33],togeth:4,toggl:33,tolstikhin:19,tool:[2,6],top:6,torch:[15,16,18,19,24,29,30,34],tr:19,track:19,train:[0,2,19,29,30,33],train_frac:[2,33],trainer:11,transform:[0,5,10,12,15],tweak:3,two:[0,6,18,19],type:[5,6,15,19],under:9,underli:[0,6],unet3d:30,unet:[3,12,20,26],unet_3d:[12,26],unet_downconv:[12,26],unet_upconv:[12,26],unfinish:[5,6],union:[15,19,33,34],uniqu:33,unnecessari:0,unpack:6,up:[1,6,30],up_conv:30,upconv:30,upconvolut:30,upsampl:33,us:[0,2,3,5,6,9,12,15,16,19,33,34],usag:31,user:[2,3],usual:5,util:[0,5,6,9,12,15,16,20,26,32],vae:[9,12,20],val:[2,33],val_frac:33,valid:33,valu:[5,6,9,19,33,34],variabl:[3,9,14,19],varianc:18,verbos:[5,6,19],veri:6,version:[1,9,12,19],version_str:9,via:9,virtualenv:1,wae:19,wai:3,want:[0,2],warn:15,we:[2,3,5,6,9,16,30],websit:1,weight:33,weight_init:[12,26],welcom:1,well:6,what:[5,6],when:[1,3,15,19,30],where:[5,6,15,33],wherev:0,whether:[5,6,15,33,34],which:[0,2,3,5,6,9,16,19,33,34],whole:6,wi:19,width:19,wise:18,within:9,without:19,work:[0,1,18],worker:5,workflow:[3,6],would:9,wrangl:[0,33],wrap:16,write:15,write_every_n_row:[5,6],x1:29,x2:29,x:[19,30,34],x_1:19,x_2:19,x_doubleconv_down:30,yaml:[0,3,5,6,11],ye:30,yield:3,you:[0,1,2,5,8,9],your:[0,1,6,8,9],your_development_typ:1,your_name_her:1,zoo:11,zoo_root:9},titles:["serotiny CLI","Contributing","Dataframe wrangling","Dynamic imports","Example workflows","Extraction features from images","Applying image transforms","serotiny","Installation","Model zoo","serotiny","Quickstart","serotiny package","serotiny.datamodules package","serotiny.imports package","serotiny.io package","serotiny.io.dataframe package","serotiny.io.dataframe.loaders package","serotiny.losses package","serotiny.metrics package","serotiny.models package","serotiny.models.callbacks package","serotiny.models.inference package","serotiny.models.unet package","serotiny.models.utils package","serotiny.models.vae package","serotiny.networks package","serotiny.networks.basic_cnn package","serotiny.networks.layers package","serotiny.networks.mlp package","serotiny.networks.unet package","serotiny.networks.utils package","serotiny.transforms package","serotiny.transforms.dataframe package","serotiny.transforms.image package","serotiny.transforms.pipeline package"],titleterms:{"class":17,"import":[3,14],abstract_load:17,align:34,appli:6,base_va:25,basic_cnn:27,bind:3,buffered_patch_dataset:15,calculate_blur:19,calculate_fid:19,call:2,callback:21,chain:2,cli:0,column:17,comment:6,conditional_va:25,condprior_va:25,config:0,configur:[5,6],content:[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],continuous_bernoulli:18,contribut:1,convolution_block:28,datafram:[2,16,17,33],dataframe_dataset:16,datamodul:13,deploi:1,dict:0,double_convolut:30,dummi:13,dynam:3,dynamic_import:14,exampl:[4,5,6],extract:5,featur:5,feature_extract:34,features_to_extract:5,from:[5,8],gaussian:13,get:1,imag:[5,6,15,34],image2d:17,image3d:17,image_va:25,incept:19,indic:7,infer:22,init:3,instal:8,invok:3,io:[15,16,17],kl_diverg:18,layer:28,load:[9,11],loader:17,loss:18,manifest_datamodul:13,metric:19,mlp:29,model:[9,11,15,20,21,22,23,24,25],modul:[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],multipl:2,network:[26,27,28,29,30,31],normal:34,optimizer_util:24,output:6,packag:[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],pad:34,pass:0,patch:13,pearson:19,pipelin:[2,6,35],plot_xhat:21,positional_arg:3,project:34,quickstart:11,quilt:15,random_imag:17,reader:16,releas:8,resiz:34,sequenti:31,serotini:[0,7,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],singl:2,sourc:8,spatial_pyramid_pool:28,specifi:6,stabl:8,start:1,submodul:[13,14,15,16,17,18,19,21,22,23,24,25,27,28,29,30,31,33,34,35],subpackag:[12,15,16,20,26,32],swap:34,tabl:7,tabular_conditional_va:25,tabular_condprior_va:25,tabular_va:25,test:2,tile_predict:22,train:[9,11],transform:[2,6,32,33,34,35],transforms_to_appli:6,unet:[23,30],unet_3d:30,unet_downconv:30,unet_upconv:30,util:[24,31,35],vae:25,weight_init:31,workflow:4,wrangl:2,zoo:9}})