%% Fig.2
% Fig.2a
% plot using GraphPad Prism 9.

% Fig.2b, c, and d
% inflated brain surface were ploted using SUMA (https://afni.nimh.nih.gov/Suma).

% axial maps (CanlabCore) 
clear;clc;close all
imgname = 'glm_NA-NV_fdr05.nii';
mask = which('gray_matter_mask.img');

img= fmri_data(imgname,mask);
cl = region(img);

slice_num = 28; % the slice you want to plot

figure; o2 = fmridisplay;
axh1 = axes('Position', [.01 .01 1 1]);
o2 = montage(o2, 'axial', 'wh_slice', [0 0 slice_num],'existing_axes', axh1);
o2 = addblobs(o2,cl,'splitcolor', {[.23 1 1], [0.17 0.61 1], [0.99 0.46 0], [1 1 0]}, ...
        'noverbose');
snapnow

%% Fig.3
% Fig.3a, b, and c
% see 
% Fig.2b, c, and d

% Fig.3d, e
% see 
% Main_analysis_scripts.m

%% Fig.4
% see
% Fig.2b, c, and d

%% Fig.5
% Fig.5a, and b 
% Script used for generating the riverplot can be obtained from 
% Čeko et al. (2021, Nature Neuroscience, https://doi.org/10.1038/s41593-022-01082-w).

% Fig.5c
% plot using GraphPad Prism 9.

% Fig.5d
% Script can be obtained from Koban et al. (2019, Nature Communications, 
% https://www.nature.com/articles/s41467-019-11934-y)

% Fig.5e and f
% see
% Fig.2b, c, and d

%% Fig.6 
% Fig.6a
% see 
% Main_analysis_scripts.m

% Fig.6b
% analysis part see
% Main_analysis_scripts.m
% plot figure
clear; clc; close all;
datadir = 'path that contained your network-based analysis reuslts';
cond = {'NAvsNV','NRvsNV','NVvsNeutV'};
c = 1;%  manually switch the contrasts
num_dat = 9;

dat = cell(1,num_dat);
for p = 1:num_dat
    if p < 8
        dat_tmp = load([datadir,cond{c},'_',num2str(p),'.mat']);
        dat{p+2}.num = dat_tmp.num_feats_within;
        dat{p+2}.acc =  dat_tmp.pred_outcome_acc;
    elseif p == 8
        dat_tmp = load([datadir,cond{c},'_',num2str(p),'.mat']);
        dat{2}.num = dat_tmp.num_feats_within;
        dat{2}.acc =  dat_tmp.pred_outcome_acc;
    elseif p == 9
        dat_tmp = load([datadir,cond{c},'_',num2str(p),'.mat']);
        dat{1}.num = dat_tmp.num_feats_within;
        dat{1}.acc =  dat_tmp.pred_outcome_acc;
    end
end
clear data_tmp

num_iterations = 1000;
num_parcels = 7;

mycolors={[0 0 0]/255 [243 146 3]/255 [1 115 178]/255  [241 76 193]/255  
    [57,182,101]/255  [18 113 28]/255  [139 43 226]/255 [232 0 11]/255 [159 72 0]/255};

x = cell(1,num_dat); y = cell(1,num_dat); 
for it=1:1000
    for p = 1:num_dat
        [~, ~,x{p}(it,:),y{p}(it,:)] = createFit(dat{p}.num(:)', ...
            dat{p}.acc(it,:),'color',mycolors{p},'linewidth',3);
    end
end
%
close all;
figure;
hold on;
for p = 1:num_dat
    boundedline(mean(x{p}),mean(y{p}), std(y{p}),'alpha', 'cmap',mycolors{p});
end

h=findobj(gca,'type','line');
set(h,'linewidth',2);
set(gca,'XScale','log')
atlas_labels={'Whole-brain','Frontal','Visual','Somatomotor','dAttention',...
    'vAttention','Limbic','Frontoparietal','Default'};

xlabel('Number of voxels', 'FontWeight', 'BOLD')
ylabel('Prediction-outcome','FontWeight', 'BOLD')
xlim([40 1000000])
set(gca,'ylim',[0.4,1], 'YTick', 0.4:0.1:1, 'LineWidth',1.5, 'FontWeight', 'BOLD')

hold on;
for p =1:num_dat
    plot(dat{p}.num, mean(dat{p}.acc), '.', 'markersize', 20, 'color',mycolors{p});
end
legend(h(end:-1:1),atlas_labels,'Location','SouthEast');
set(gcf,'unit','centimeters','position',[10 10 20 20]);

% Fig.6c and d
% see
% Main_analysis_scripts.m

% Fig.6e
% see
% Figure_permutation_test.py

%% -----------------------------------------------------------------------
% Figures in the supplementary
%-------------------------------------------------------------------------
%% Fig.1
% Fig.1a, b, and c
% see
% Fig.2b, c, and d

% Fig.1d
% see 
% Figure_permutation_test.py