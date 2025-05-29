%% Main analysis scripts
%% Mass-univariate analysis
% scripts can be obtained from https://github.com/zhou-feng/matlab_scripts
%% Predict on discovery cohort
% the discovery dataset is availabe at 
% some scripts we used were modified based on the code provided by 
% Kohoutová et al.(2020, Nature Protocols, 
% https://www.nature.com/articles/s41596-019-0289-5)

clear; close all
datadir = 'your 1st level anaylsis dir';
maskdir = 'your 2nd level anaylsis dir of one codition';
mask = [maskdir,'/2nd_level_analysis/mask.nii'];


% load images.
cd(datadir)
nsub = 59;
con1_imgs = filenames(fullfile(datadir, '/sub*/con_000X.nii'));% cond1
con2_imgs = filenames(fullfile(datadir, '/sub*/con_000X.nii'));% cond2

data = fmri_data([con1_imgs; con2_imgs],mask);
data = rescale(data,'zscoreimages');

% Prepare and add outcome variable into dat.Y.
data.Y = [ones(size(con1_imgs,1),1); -ones(size(con2_imgs,1),1)]; % cond1: 1, cond2: -1

% Define the training and test sets for cross-validation.
ntrial1 = size(con1_imgs,1)/nsub;
ntrial2 = size(con2_imgs,1)/nsub;
n_folds = [repmat(1:nsub, ntrial1,1) repmat(1:nsub, ntrial2,1)];
n_folds = n_folds(:);

% Fit an SVM model to the training data.
[~, stats_loso] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', ...
    n_folds, 'error_type', 'mcr');

% Predict preformance
ROC_loso = roc_plot(stats_loso.dist_from_hyperplane_xval, data.Y == 1, ...
    'threshold', 0,'color',[255 194 75]);
%% Bootstrap test
[~, stats_boot] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', 1, ...
    'error_type', 'mcr', 'bootweights', 'bootsamples', 10000);
data_thresh = threshold(stats_boot.weight_obj, .05, 'fdr');

%% Encoding model estimate
% scripts can be obtained from Zhou et al.(2021, Nature Communications, 
% https://www.nature.com/articles/s41467-021-26977-3 and
% https://github.com/zhou-feng/fMRI-studies/tree/main/Fear_experience_signature)

%% Test performance on validation cohort
clear; clc; close all

decoderdir = 'path of the decoder';
datadir = 'path of validation cohort data';

mask = which('gray_matter_mask.img');

NERS_A = [decoderdir,'/weights_boot_NERS-A.nii'];
NERS_R = [decoderdir,'/weights_boot_NERS-R.nii'];
NNES = [decoderdir,'/weights_boot_NNES.nii'];

cd (datadir)
ntrial = 1; ncontrast = 4; 
cont_imgs{1} = filenames(fullfile(datadir,'/sub-*/con_0001.nii'), 'char');% NeutV
cont_imgs{2} = filenames(fullfile(datadir,'/sub-*/con_0002.nii'), 'char');% NV
cont_imgs{3} = filenames(fullfile(datadir,'/sub-*/con_0003.nii'), 'char');% NA
cont_imgs{4} = filenames(fullfile(datadir,'/sub-*/con_0004.nii'), 'char');% NR
nsub = size(cont_imgs{1},1);

data_v = fmri_data(cont_imgs, mask);
data_v = rescale(data_v,'zscoreimages'); 

pexp_NERS_A = apply_mask(data_v, NERS_A, 'pattern_expression', 'ignore_missing');
pexp_NERS_R = apply_mask(data_v, NERS_R, 'pattern_expression', 'ignore_missing');
pexp_NNES = apply_mask(data_v, NNES, 'pattern_expression', 'ignore_missing');

% reshape pexp values to have differnt conditions in different columns
pexp_NERS_A = reshape(pexp_NERS_A, nsub*ntrial, ncontrast);
pexp_NERS_R = reshape(pexp_NERS_R, nsub*ntrial, ncontrast);
pexp_NNES = reshape(pexp_NNES, nsub*ntrial, ncontrast);

x = linspace(-1, 1, 100); y = x; 
rgb_color = {[255 194 75],[246 111 105],[21 151 165],[14 96 107]};
ncolor = size(rgb_color,2);
colorset = cell(1, ncolor);
for j= 1:ncolor
    colorset{j} = reshape((dec2hex(rgb_color{j})).', 1, []);
end

labelname1 = 'NERS-A for NA vs. NV';
roc_1 = roc_plot([pexp_NERS_A(:,3);pexp_NERS_A(:,2)], [true(nsub*ntrial,1); ...
    false(nsub*ntrial,1)], 'twochoice','color',['#',colorset{1}]);
if roc_1.accuracy_p < 0.001; P = '< 0.001'; 
else; P = ['= ', pval]; end
acc_se1 = ['ACC = ',num2str(round(roc_1.accuracy*100)),'% ± ', ...
    num2str(round(roc_1.accuracy_se*100,1)),'%, p ',P];
hold on

labelname2 = 'NERS-R for NR vs. NV';
roc_2 = roc_plot([pexp_NERS_R(:,4);pexp_NERS_R(:,2)], [true(nsub*ntrial,1); ...
    false(nsub*ntrial,1)], 'twochoice','color',['#',colorset{2}]);
if roc_2.accuracy_p < 0.001; P = '< 0.001'; 
else; P = ['= ', pval]; end
acc_se2 = ['ACC = ',num2str(round(roc_2.accuracy*100)),'% ± ', ...
    num2str(round(roc_2.accuracy_se*100,1)),'%, p ',P];
hold on

labelname3 = 'NNES for NV vs. NeutV';
roc_3 = roc_plot([pexp_NNES(:,2);pexp_NNES(:,1)], [true(nsub,1);false(nsub,1)], ...
    'twochoice','color',['#',colorset{3}]);
if roc_3.accuracy_p < 0.001; P = '< 0.001'; 
else; P = ['= ', pval]; end
acc_se3 = ['ACC = ',num2str(round(roc_3.accuracy*100)),'% ± ', ...
    num2str(round(roc_3.accuracy_se*100,1)),'%, p ',P];
hold on

plot(x, y, '--', 'Color', [.5, .5, .5], 'LineWidth', 1);
hold off

set(gcf,'unit','centimeters','position',[1 1 26 25]);
set(gca,'Position',[.175 .175 .75 .75],'box','off','TickDir','out', ...
    'XLim',[-0.02,1],'YLim',[0,1]);
legend({acc_se1,labelname1,acc_se2,labelname2,acc_se3,labelname3,acc_se4,labelname4}, ...
    'Fontsize', 18,'Position',[0.5,0.25,0.40,0.15],'box','off');

% Violin plots are draw used a function from cocoanCORE, 
% https://github.com/cocoanlab/cocoanCORE 
close all
c = 1; % 2, 3 % manully change the contrast 
if c == 1
    contrast = 'NAvsNV'; labels = {'NA','NV'};
    colorindex = [1,4];
    signature_response1 = pexp_NERS_A(:,3);
    signature_response2 = pexp_NERS_A(:,2);
elseif c == 2
    contrast = 'NRvsNV'; labels = {'NR','NV'};
    colorindex = [2,4];
    signature_response1 = pexp_NERS_R(:,4);
    signature_response2 = pexp_NERS_R(:,2);
elseif c == 3
    contrast = 'NVvsNeutV'; labels = {'NV','NeutV'};
    colorindex = [4,3];
    signature_response1 = pexp_NNES(:,2);
    signature_response2 = pexp_NNES(:,1);
end

figure;
out = plot_specificity_box(signature_response1, signature_response2, ...
    'colors', colorset(colorindex,:));
ylabel('Signature response', 'fontsize', 18); 
xticklabels(labels); xtickangle(45);
set(gcf,'unit','centimeters','position',[10 10 20 20]);
set(gca, 'fontsize', 18, 'linewidth', 2,'ylim', [-4 8]); % change 'ylim' based on your data

%% Network-based prediction
% The scripts of this section are modified based on evaluate_spatial.m from CanlabCore.
clear;clc;
% pool = parpool(4);
basedir = 'path to your data root';
savedir = [basedir,'/results/network_predict/'];cd(savedir)
atlasdir = [basedir,'/data/atlas/001_network_predict/'];
num_atlas = 8; % manually change atlas file
img = filenames(fullfile([atlasdir,num2str(num_atlas),'*.nii']),'char');
atlas_obj = fmri_data(img);

cond = {'NAvsNV','NRvsNV','NVvsNeutV'}; % contrasts that you want to run 
num_voxs = [50 150 250 500 750 1000 2000 4000 6000 8000 10000 14000 ...
    18000 25000 60000 120000];

for c = 1:3
    load([basedir,'/data/dat_obj/Discovery_',cond{c},'.mat'])
    [num_feats_within, pred_outcome_acc] = ...
    evaluate_spatial_scale_parcel_svm(data,atlas_obj,'cv_svm', 'nfolds', ...
    n_folds,'num_voxels',num_voxs,'verbose',0,'atlas',num_atlas,'cond',c);
    save ([savedir,cond{c},'_',num2str(num_atlas),'.mat'],...
        'num_feats_within','pred_outcome_acc','-mat')
end

%% Searchlight-based prediction
% see searchlight_dream.m for more detials
clear;clc;
basedir = 'path to your data root';

contrast = 'NAvsNV'; maskfolder = 'NA-NV'; % Manually switching conditions
% contrast = 'NRvsNV'; maskfolder = 'NR-NV';
% contrast = 'NVvsNeutV'; maskfolder = 'NV-NeutV';

load ([basedir,'/data/dat_obj/Discovery_',contrast,'.mat'])
mask = ['2nd level analysis dir',maskfolder,'/mask.nii'];
data = trim_mask(data);
%
dist_n = 10; % The number of jobs (brain chunks) you want to create
r = 3;
modeldir = [basedir,'results/searchlight/r',num2str(r),'/',contrast];
if ~exist(modeldir,'dir'); mkdir (modeldir); end

holdout_set = n_folds;
searchlight_dream(data,dist_n,mask,'algorithm_name','cv_svm','r', r, ...
    'modeldir',modeldir,'cv_assign',holdout_set, ...
    'save_weights','outcome_method', 'twochoice');
cd (modeldir)

script_filenames = filenames(fullfile([modeldir,'/searchlight_*.m']));
pool = parpool(10);

parfor k = 1:length(script_filenames)
    runFile(cell2mat(script_filenames(k)));
end

searchlight_saveresults(modeldir);

%% Permutation test
%% Test on validation cohort
clear; clc; close all
decoderdir = 'decoder weighted maps dir';
datadir = 'your data dir';
savedir = 'path to save your data';
mask = which('gray_matter_mask.nii');
currentdir = pwd;
cd (datadir)

decoder1 = [decoderdir,'decoder weighted maps name.nii'];
decoder2 = [decoderdir,'decoder weighted maps name.nii'];
decoder3 = [decoderdir,'decoder weighted maps name.nii'];

norm = 'zscoreimages';

ntrial = 1; ncontrast = 4; 
cont_imgs1{1} = filenames(fullfile(datadir,'/sub-1*/con_000X.nii'), 'char');% NeutV
cont_imgs1{2} = filenames(fullfile(datadir,'/sub-1*/con_000X.nii'), 'char');% NV
cont_imgs1{3} = filenames(fullfile(datadir,'/sub-1*/con_000X.nii'), 'char');% NA
cont_imgs1{4} = filenames(fullfile(datadir,'/sub-1*/con_000X.nii'), 'char');% NR
%
data_test1 = fmri_data(cont_imgs, mask);
nsub_hc = size(cont_imgs{1},1);

% load HC data first
dat_temp_hc = data_test1.dat;
indx_hc = repmat(1:nsub_hc, 1, ncontrast);
%%
clear data_test1

% then load CU data
data_test1 = fmri_data(cont_imgs1, mask);%%, mask
nsub_cu= size(cont_imgs1{1},1);

dat_temp_cu = data_test1.dat;
indx_cu = repmat(100+1:100+nsub_cu, 1, ncontrast);

dat_all = [dat_temp_hc,dat_temp_cu];
data_test = data_test1;
data_test.dat = [];


indx_all = [indx_hc,indx_cu];
indx_perm = [repmat(1:nsub_hc, 1, 1),repmat(100+1:100+nsub_cu, 1, 1);];

num_indx = size(indx_perm, 2);
%%
npermu = 10000;
ACC_perm = zeros(npermu,3);
p_perm = zeros(npermu,3);
h=waitbar(0,'Please Wait');%Progress bar
for n = 1: npermu
    indx_random = indx_perm(randperm(num_indx));
    indx_shuffled = ismember(indx_all,indx_random(1:nsub_cu));

    data_test.dat = dat_all(:,indx_shuffled);
    data_test = rescale(data_test,norm);

    pexp_decoder1 = apply_mask(data_test, decoder1, 'pattern_expression', 'ignore_missing');
    pexp_decoder2 = apply_mask(data_test, decoder2, 'pattern_expression', 'ignore_missing');
    pexp_decoder3 = apply_mask(data_test, decoder3, 'pattern_expression', 'ignore_missing');

    % reshape pexp values to have differnt conditions in different columns
    pexp_decoder1 = reshape(pexp_decoder1, nsub_cu*ntrial, ncontrast);
    pexp_decoder2 = reshape(pexp_decoder2, nsub_cu*ntrial, ncontrast);
    pexp_decoder3 = reshape(pexp_decoder3, nsub_cu*ntrial, ncontrast);

    roc_1 = roc_plot([pexp_decoder1(:,3);pexp_decoder1(:,2)], ...
        [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
        'twochoice','noplot','nooutput');
    ACC_perm(n,1) = roc_1.accuracy;
    p_perm(n,1) = roc_1.accuracy_p;

    roc_2 = roc_plot([pexp_decoder2(:,4);pexp_decoder2(:,2)], ...
        [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
        'twochoice','noplot','nooutput');
    ACC_perm(n,2) = roc_2.accuracy;
    p_perm(n,2) = roc_2.accuracy_p;

    roc_3 = roc_plot([pexp_decoder3(:,2);pexp_decoder3(:,1)], ...
        [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
        'twochoice','noplot','nooutput');
    ACC_perm(n,3) = roc_3.accuracy;
    p_perm(n,3) = roc_3.accuracy_p;

    str=['Processing permutation -',num2str(n),' now'];%Progress bar
    waitbar(n/npermu,h,str)%Progress bar
end

% get orginal ACC of HC
data_hc = data_test; data_hc.dat = dat_temp_hc;
data_hc = rescale(data_hc,norm); 
% ---------------------------------------------------------------------
% reshape pexp values to have differnt conditions in different columns
pexp_decoder1 = reshape(pexp_decoder1, nsub_hc*ntrial, ncontrast);
pexp_decoder2 = reshape(pexp_decoder2, nsub_hc*ntrial, ncontrast);
pexp_decoder3 = reshape(pexp_decoder3, nsub_hc*ntrial, ncontrast);
% ----------------------------------------------------------------------
roc_1 = roc_plot([pexp_decoder1(:,3);pexp_decoder1(:,2)], ...
    [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
    'twochoice','noplot','nooutput');
ACC_og(2,1) = roc_1.accuracy;

roc_2 = roc_plot([pexp_decoder2(:,4);pexp_decoder2(:,2)], ...
    [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
    'twochoice','noplot','nooutput');
ACC_og(2,2) = roc_2.accuracy;

roc_3 = roc_plot([pexp_decoder3(:,2);pexp_decoder3(:,1)], ...
    [true(nsub_cu*ntrial,1);false(nsub_cu*ntrial,1)], ...
    'twochoice','noplot','nooutput');
ACC_og(2,3) = roc_3.accuracy;
% get p value for permutation test
P_val(1,1) = sum(ACC_perm(:,1) >= ACC_og(1,1))/npermu;
P_val(1,2) = sum(ACC_perm(:,2) >= ACC_og(1,2))/npermu;
P_val(1,3) = sum(ACC_perm(:,3) >= ACC_og(1,3))/npermu;

save([savedir,'/permu_CU&HC.mat'],'ACC_perm','p_perm','ACC_og','P_val','-mat'); 

T = table(ACC_perm(:, 1), ACC_perm(:, 2), ACC_perm(:, 3), 'VariableNames', ...
    {'decoder1', 'decoder2', 'decoder3'});
writetable(T,[savedir,'\permu_CU&HC_og.csv'])