function [net, info] = cnn_train_audio_mr(net, data_f, data_m, getBatch, getValid,  varargin)
% NMF_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.fixedLayers = [];
opts.validFreq = 5 ;
opts.startSave = 150;

opts.train = [] ;
opts.val = [] ;
opts.train2 = [] ;
opts.val2 = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = true ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = 'data/exp' ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts.C = 1;
opts.use_single = 1;
opts.epsilon = 1e-8;
opts = vl_argparse(opts, varargin) ;

training_proportion = 0.95;

if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for ii=1:size(net,2)
for i=1:numel(net{ii}.layers)
    if ~strcmp(net{ii}.layers{i}.type,'conv') && ~strcmp(net{ii}.layers{i}.type,'nmf'), continue; end
    
    if strcmp(net{ii}.layers{i}.type,'conv') ,
        net{ii}.layers{i}.filtersMomentum = zeros('like',net{ii}.layers{i}.filters) ;
        net{ii}.layers{i}.biasesMomentum = zeros('like',net{ii}.layers{i}.biases) ;
    end
    
    
    if strcmp(net{ii}.layers{i}.type,'nmf') ,
        net{ii}.layers{i}.D1Momentum = zeros('like',net{ii}.layers{i}.D1) ;
        net{ii}.layers{i}.D2Momentum = zeros('like',net{ii}.layers{i}.D2) ;
    end
    
    if ~isfield(net{ii}.layers{i}, 'filtersLearningRate')
        net{ii}.layers{i}.filtersLearningRate = 1 ;
    end
    if ~isfield(net{ii}.layers{i}, 'biasesLearningRate')
        net{ii}.layers{i}.biasesLearningRate = 1 ;
    end
    if ~isfield(net{ii}.layers{i}, 'filtersWeightDecay')
        net{ii}.layers{i}.filtersWeightDecay = 1 ;
    end
    if ~isfield(net{ii}.layers{i}, 'biasesWeightDecay')
        net{ii}.layers{i}.biasesWeightDecay = 1 ;
    end
end
end
for ii=1:size(net,2)
if opts.useGpu
    net{ii} = vl_simplenn_move(net{ii}, 'gpu') ;
    for i=1:numel(net{ii}.layers)
        if ~strcmp(net{ii}.layers{i}.type,'conv'), continue; end
        net{ii}.layers{i}.filtersMomentum = gpuArray(net{ii}.layers{i}.filtersMomentum) ;
        net{ii}.layers{i}.biasesMomentum = gpuArray(net{ii}.layers{i}.biasesMomentum) ;
    end
end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.NSDR = [] ;
info.val.stat = [] ;

max_SDR = 0;

info.val.stat{end+1} = [] ;
info.val.NSDR(end+1) = 0 ;

modelPath = fullfile(opts.expDir, 'net-epoch-0.mat') ;

%

%C = size(imdb.images.data,2);

lr = 0 ;
res = [] ;
for epoch=1:opts.numEpochs
 %	imdb_f = prepareData_matconvnet(data_f,opts.C,'female',opts.use_single,0);
%	imdb_m = prepareData_matconvnet(data_m,opts.C,'male',opts.use_single,0);
%	if isempty(opts.train), opts.train = find(imdb_f.images.set==1) ; end
%	if isempty(opts.train2), opts.train2 = find(imdb_m.images.set==1) ; end
%	if isempty(opts.val), opts.val = find(imdb_f.images.set==2) ; end
%	if isempty(opts.val2), opts.val2 = find(imdb_m.images.set==2) ; end

    prevLr = lr ;
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

    % fast-forward to where we stopped
    modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
    modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
    if opts.continue
        if exist(sprintf(modelPath, epoch),'file'), continue ; end
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(sprintf(modelPath, epoch-1), 'net', 'info') ;
        end
    end
    
    %train = opts.train(randperm(numel(opts.train))) ;
    %train2 = opts.train2(randperm(numel(opts.train2))) ;
    %val = opts.val ;
    %val2 = opts.val2 ;
    
    info.train.objective(end+1) = 0 ;
    info.train.error(end+1) = 0 ;
    info.train.topFiveError(end+1) = 0 ;
    info.train.speed(end+1) = 0 ;
    info.val.NSDR(end+1) = 0 ;
    info.val.stat{end+1} = [] ;
    info.val.objective(end+1) = 0 ;
    

    % reset momentum if needed
    if prevLr ~= lr
        fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
	for ii=1:size(net{ii},2)
        for l=1:numel(net{ii}.layers)
            if ~strcmp(net{ii}.layers{l}.type, 'conv'), continue ; end
            net{ii}.layers{l}.filtersMomentum = 0 * net{ii}.layers{l}.filtersMomentum ;
            net{ii}.layers{l}.biasesMomentum = 0 * net{ii}.layers{l}.biasesMomentum ;
        end
	end
    end
    
	nframes_train_1 = round(training_proportion * size(data_f.X,2));
	nframes_train_2 = round(training_proportion * size(data_m.X,2));

	train1 = round(opts.C/2) + randperm(nframes_train_1-opts.C);
	train2 = round(opts.C/2) + randperm(nframes_train_2-opts.C);
	train1 = train1(1:min(nframes_train_1, nframes_train_2)-opts.C);	
	train2 = train2(1:min(nframes_train_1, nframes_train_2)-opts.C);	

	if epoch==1
	nframes_val_1 = size(data_f.X,2) - nframes_train_1 ;
	nframes_val_2 = size(data_m.X,2) - nframes_train_2 ;
	val1 = length(train1) + round(opts.C/2) + randperm(nframes_val_1-opts.C);
	val2 = length(train1) + round(opts.C/2) + randperm(nframes_val_2-opts.C);
	val1 = val1(1:min(nframes_val_1, nframes_val_2)-opts.C);	
	val2 = val2(1:min(nframes_val_1, nframes_val_2)-opts.C);	
	end

	%input has 4 dimensions: 1 x C x F x bs


	N = length(train1);
	im = zeros(1, opts.C, size(data_f.X,1), opts.batchSize, 'single','gpuArray');
	im_mix = zeros(1, opts.C, size(data_f.X,1), opts.batchSize, 'single','gpuArray');
	im1 = zeros(1, opts.C, size(data_f.X,1), opts.batchSize, 'single','gpuArray');
	im2 = zeros(1, opts.C, size(data_f.X,1), opts.batchSize, 'single','gpuArray');

	for t=1:opts.batchSize:N-opts.batchSize+1
		I1 = train1(t:t+opts.batchSize-1);
		I2 = train2(t:t+opts.batchSize-1);
		for c=-floor(opts.C/2):floor(opts.C/2)
			im1(1,c+floor(opts.C/2)+1,:,:) = data_f.X(:,I1+c);	
			im2(1,c+floor(opts.C/2)+1,:,:) = data_m.X(:,I2+c);	
		end
		im_mix = im1+im2;
		im = abs(im_mix);
		tmp = sqrt(sum(sum(im.^2,2),3));
		im = im./repmat(opts.epsilon + tmp, [1 opts.C size(im,3) 1]) ;
	
    
    %N = min(numel(train),numel(train2));
	%for t=1:opts.batchSize:N
        
        % get next image batch and labels
      %  batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
      %  batch2 = train2(t:min(t+opts.batchSize-1, numel(train2))) ;
        
      %  if length(batch)~=length(batch2)
      %      m = min(length(batch),length(batch2));
      %      batch = batch(1:m);
      %      batch2 = batch2(1:m);
      %  end
        
        
        batch_time = tic ;
        fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(N/opts.batchSize)) ;
        
       % [im,im_mix, im1,im2] = getBatch(imdb_f, imdb_m, batch,batch2) ;
        
       % if opts.useGpu
       %     im = gpuArray(im) ;
       %     im1 = gpuArray(im1);
       %     im2 = gpuArray(im2);
       %     im_mix = gpuArray(im_mix);
       % end
        
 
        % fprop and backprop
%	if t==1
%	size(im)
%	size(im1)
%	size(im2)
%	keyboard
%	end
	net{end}.layers{end}.Ymix = im_mix(:,ceil(opts.C/2),:,:);
	net{end}.layers{end}.Y1 = im1(:,ceil(opts.C/2),:,:);
	net{end}.layers{end}.Y2 = im2(:,ceil(opts.C/2),:,:);
	res = vl_multiresnn(net,im,one,[], ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync);
	        
        batch_time = toc(batch_time) ;
        % gradient step
	for ii=1:size(net,2)
        for l=1:numel(net{ii}.layers)
            
            if sum( opts.fixedLayers == l) == 1, continue; end
                   
            if ~strcmp(net{ii}.layers{l}.type, 'conv'), continue ; end

            net{ii}.layers{l}.filtersMomentum = ...
                opts.momentum * net{ii}.layers{l}.filtersMomentum ...
                - (lr * net{ii}.layers{l}.filtersLearningRate) * ...
                (opts.weightDecay * net{ii}.layers{l}.filtersWeightDecay) * net{ii}.layers{l}.filters ...
                - (lr * net{ii}.layers{l}.filtersLearningRate) / opts.C / opts.batchSize * res{ii}(l).dzdw{1} ;
            
            net{ii}.layers{l}.biasesMomentum = ...
                opts.momentum * net{ii}.layers{l}.biasesMomentum ...
                - (lr * net{ii}.layers{l}.biasesLearningRate) * ....
                (opts.weightDecay * net{ii}.layers{l}.biasesWeightDecay) * net{ii}.layers{l}.biases ...
                - (lr * net{ii}.layers{l}.biasesLearningRate) / opts.C / opts.batchSize * res{ii}(l).dzdw{2} ;
            
            net{ii}.layers{l}.filters = net{ii}.layers{l}.filters + net{ii}.layers{l}.filtersMomentum ;
            net{ii}.layers{l}.biases = net{ii}.layers{l}.biases + net{ii}.layers{l}.biasesMomentum ;
        end
        end
        % print information
        speed = opts.batchSize/batch_time ;
        
        info.train.objective(end) = info.train.objective(end) + sum(double(gather(res{end}(end).x))) ;
        info.train.speed(end) = info.train.speed(end) + speed ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        n = t + opts.batchSize - 1 ;
        fprintf(' obj %.4f', ...
            info.train.objective(end)/n) ;
        
        fprintf('\n') ;
        % debug info
        if opts.plotDiagnostics
		error('not implemented yet')
            figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
        end
    end % next batch
    
    %----------------------------------------------------------------------
	if 1
   
	%validation using the same cost function
	for t=1:opts.batchSize:length(val1)-opts.batchSize+1
		I1 = val1(t:t+opts.batchSize-1);
		I2 = val2(t:t+opts.batchSize-1);
		for c=-floor(opts.C/2):floor(opts.C/2)
			im1(1,c+floor(opts.C/2)+1,:,:) = data_f.X(:,I1+c);	
			im2(1,c+floor(opts.C/2)+1,:,:) = data_m.X(:,I2+c);	
		end
		im_mix = im1+im2;
		im = abs(im_mix);
		tmp = sqrt(sum(sum(im.^2,2),3));
		im = im./repmat(opts.epsilon + tmp, [1 opts.C size(im,3) 1]) ;
        
        batch_time = tic ;
        fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(N/opts.batchSize)) ;
 
        % fprop and backprop
	net{end}.layers{end}.Ymix = im_mix(:,ceil(opts.C/2),:,:);
	net{end}.layers{end}.Y1 = im1(:,ceil(opts.C/2),:,:);
	net{end}.layers{end}.Y2 = im2(:,ceil(opts.C/2),:,:);
	res = vl_multiresnn(net,im,[],[], ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync);
 
        info.val.objective(end) = info.val.objective(end) + sum(double(gather(res{end}(end).x))) ;
        n = t + opts.batchSize - 1 ;
        fprintf(' obj %.4f', ...
            info.val.objective(end)/n) ;
        fprintf('\n') ;

	end	

	end
    % save
    info.train.objective(end) = info.train.objective(end) / numel(train1)/opts.C ;
    info.train.speed(end) = numel(train1) / info.train.speed(end)/opts.C ;
%     info.val.objective(end) = info.val.objective(end) / numel(val) ;
%     info.val.speed(end) = numel(val) / info.val.speed(end) ;
%     save(sprintf(modelPath,epoch), 'net', 'info') ;
%     
%     fprintf('\n') ;
%     fprintf('Objective function - Training: %.2f s, Validation: %.1f.', info.train.objective(end), info.val.objective(end)) ;
%     fprintf('\n') ;
    

    if  ~mod(epoch, opts.validFreq )

    output = getValid(net);
    fprintf('Validation: \n')
    disp(output.stat)
    
    info.val.NSDR(end) = output.stat.mean_NSDR;
    info.val.stat{end} = output.stat;
    end
    
    if epoch>opts.startSave 
        if  ~mod(epoch, opts.validFreq ) %|| max_SDR < output.stat.mean_NSDR
            fprintf('saving information for epoch ... ')
            save(sprintf(modelPath,epoch), 'net', 'info') ;
        end
    end
    

	fprintf('done \n')
end





