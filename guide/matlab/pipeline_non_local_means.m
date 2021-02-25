%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%
  
  clear all %#ok
  close all
  
  
  %% Import Test Image
  
    img = [0.101961 0.0862745 0.054902 0.113725 0.298039 0.411765 0.396078 0.211765 0.156863 0.266667 0.235294 0.0862745 0.0509804 0.105882 0.0862745 0.105882;
0.0156863 0.00392157 0.054902 0.462745 0.729412 0.8 0.827451 0.823529 0.87451 0.905882 0.760784 0.478431 0.14902 0 0.0509804 0.0156863;
0.0666667 0 0.403922 0.705882 0.345098 0.207843 0.258824 0.411765 0.486275 0.388235 0.529412 0.8 0.6 0.113725 0 0.0862745;
0.0509804 0.145098 0.686275 0.372549 0 0 0 0 0 0.176471 0.160784 0.317647 0.776471 0.486275 0.0392157 0.054902;
0.0509804 0.368627 0.662745 0.0666667 0.0352941 0.0392157 0.133333 0.513726 0.443137 0.639216 0.419608 0 0.454902 0.772549 0.313726 0.0509804;
0.0823529 0.619608 0.482353 0.00392157 0.0627451 0 0.356863 0.968627 0.87451 0.388235 0.0588235 0.0156863 0.133333 0.643137 0.576471 0.196078;
0.235294 0.67451 0.239216 0 0.113725 0.156863 0.537255 0.827451 0.913725 0.454902 0.0117647 0.0588235 0.0705882 0.537255 0.670588 0.380392;
0.435294 0.564706 0.121569 0.0156863 0.0666667 0.32549 0.72549 0.803922 0.839216 0.713726 0.262745 0.027451 0.0352941 0.47451 0.72549 0.541176;
0.556863 0.568627 0.133333 0.0235294 0.121569 0.301961 0.172549 0.345098 0.552941 1 0.470588 0 0.00784314 0.486275 0.764706 0.631373;
0.6 0.654902 0.211765 0.00784314 0.117647 0.0823529 0 0.286275 0.694118 0.788235 0.329412 0.0196078 0.0509804 0.619608 0.792157 0.643137;
0.658824 0.784314 0.486275 0 0.054902 0 0.290196 0.898039 0.901961 0.364706 0.0117647 0.0156863 0.188235 0.811765 0.847059 0.572549;
0.505882 0.756863 0.796078 0.25098 0 0 0.737255 1 0.560784 0.0392157 0.0666667 0 0.431373 0.866667 0.796078 0.384314;
0.258824 0.666667 0.823529 0.709804 0.156863 0.435294 0.972549 0.964706 0.752941 0 0 0.219608 0.780392 0.803922 0.713726 0.192157;
0.0705882 0.419608 0.67451 0.752941 0.72549 0.807843 0.831373 0.768627 0.913725 0.733333 0.560784 0.72549 0.792157 0.741176 0.341176 0.0509804;
0 0.0352941 0.321569 0.560784 0.74902 0.784314 0.772549 0.768627 0.768627 0.933333 1 0.823529 0.52549 0.286275 0 0;
0.101961 0.0509804 0.0509804 0.156863 0.317647 0.45098 0.52549 0.556863 0.572549 0.552941 0.47451 0.341176 0.172549 0.0509804 0.0509804 0.105882];



  %% PARAMETERS
  
  % input image
  pathImg   = '../data/house.mat';
  strImgVar = 'house';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [5 5];
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  %ioImg = matfile( pathImg );
  %I     = ioImg.(strImgVar);
  I = img;
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  
  %% NON LOCAL MEANS
  
  tic;
  If = nonLocalMeans( J, patchSize, filtSigma, patchSigma );
  toc
  
  %% VISUALIZE RESULT
  
  figure('Name', 'Filtered image');
  imagesc(If); axis image;
  colormap gray;
  
  figure('Name', 'Residual');
  imagesc(If-J); axis image;
  colormap gray;
  
  %% (END)

  fprintf('...end %s...\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
