function [net, info] = inv_cnn_train(net, imdb, imdb2, getBatch, varargin)
% NMF_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.train = [] ;
opts.val = [] ;
opts.train2 = [] ;
opts.val2 = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
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
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.train2), opts.train2 = find(imdb2.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isempty(opts.val2), opts.val2 = find(imdb2.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv') && ~strcmp(net.layers{i}.type,'nmf'), continue; end
    
    if strcmp(net.layers{i}.type,'conv') ,
        net.layers{i}.filtersMomentum = zeros('like',net.layers{i}.filters) ;
        net.layers{i}.biasesMomentum = zeros('like',net.layers{i}.biases) ;
    end
    
    
    if strcmp(net.layers{i}.type,'nmf') ,
        net.layers{i}.D1Momentum = zeros('like',net.layers{i}.D1) ;
        net.layers{i}.D2Momentum = zeros('like',net.layers{i}.D2) ;
    end
    
    if ~isfield(net.layers{i}, 'filtersLearningRate')
        net.layers{i}.filtersLearningRate = 1 ;
    end
    if ~isfield(net.layers{i}, 'biasesLearningRate')
        net.layers{i}.biasesLearningRate = 1 ;
    end
    if ~isfield(net.layers{i}, 'filtersWeightDecay')
        net.layers{i}.filtersWeightDecay = 1 ;
    end
    if ~isfield(net.layers{i}, 'biasesWeightDecay')
        net.layers{i}.biasesWeightDecay = 1 ;
    end
end

if opts.useGpu
    net = vl_simplenn_move(net, 'gpu') ;
    for i=1:numel(net.layers)
        if ~strcmp(net.layers{i}.type,'conv'), continue; end
        net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
        net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
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
info.val.topFiveError = [] ;
info.val.speed = [] ;

net0 = net;

val = opts.val ;
val2 = opts.val2 ;
res = [];

info.val.objective(end+1) = 0 ;
info.val.speed(end+1) = 0 ;

modelPath = fullfile(opts.expDir, 'net-epoch-0.mat') ;

% evaluation on validation set
numel(val)
for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val)));
    batch2 = val2(t:min(t+opts.batchSize-1, numel(val)));
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', 0, ...
        fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    
    [im,im_mix, im1,im2] = getBatch(imdb, imdb2, batch,batch2) ;
    
    if opts.useGpu
        im = gpuArray(im) ;
    end
    
    net.layers{end}.Ymix = im_mix;
    net.layers{end}.Y1 = im1;
    net.layers{end}.Y2 = im2;
    res = vl_simplenn(net, im, [], res, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync) ;
    
    % print information
    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time ;
    
    info.val.objective(end) = info.val.objective(end) + sum(double(gather(res(end).x))) ;
    info.val.speed(end) = info.val.speed(end) + speed ;
    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    fprintf('\n') ;
end

% save

info.val.objective(end) = info.val.objective(end) / numel(val) ;
info.val.speed(end) = numel(val) / info.val.speed(end) ;
save(sprintf(modelPath,0), 'net', 'info') ;



lr = 0 ;
res = [] ;
for epoch=1:opts.numEpochs

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
    
    train = opts.train(randperm(numel(opts.train))) ;
    train2 = opts.train2(randperm(numel(opts.train2))) ;
    val = opts.val ;
    val2 = opts.val2 ;
    
    info.train.objective(end+1) = 0 ;
    info.train.error(end+1) = 0 ;
    info.train.topFiveError(end+1) = 0 ;
    info.train.speed(end+1) = 0 ;
    info.val.objective(end+1) = 0 ;
    info.val.speed(end+1) = 0 ;
    

    % reset momentum if needed
    if prevLr ~= lr
        fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
        for l=1:numel(net.layers)
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
            net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
            net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
        end
    end
    
    
    N = min(numel(train),numel(train2));
    for t=1:opts.batchSize:N
        
        % get next image batch and labels
        batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
        batch2 = train2(t:min(t+opts.batchSize-1, numel(train2))) ;
        
        if length(batch)~=length(batch2)
            m = min(length(batch),length(batch2));
            batch = batch(1:m);
            batch2 = batch2(1:m);
        end
        
        
        batch_time = tic ;
        fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(N/opts.batchSize)) ;
        
        %[im, labels] = getBatch(imdb, batch) ;
        [im,im_mix, im1,im2] = getBatch(imdb, imdb2, batch,batch2) ;
        
        %     if opts.prefetch
        %       nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
        %       getBatch(imdb, nextBatch) ;
        %     end
        if opts.useGpu
            im = gpuArray(im) ;
        end
        
        % backprop
        net.layers{end}.Ymix = im_mix;
        net.layers{end}.Y1 = im1;
        net.layers{end}.Y2 = im2;
        
        res = vl_simplenn(net, im, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        
        % gradient step
        for l=1:numel(net.layers)
            
            
            if strcmp(net.layers{l}.type, 'nmf'),
                net.layers{l}.D1Momentum = ...
                    opts.momentum * net.layers{l}.D1Momentum ...
                    - (lr * net.layers{l}.filtersLearningRate) * ...
                    (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.D1 ...
                    - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;
                
                net.layers{l}.D2Momentum = ...
                    opts.momentum * net.layers{l}.D2Momentum ...
                    - (lr * net.layers{l}.filtersLearningRate) * ...
                    (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.D2 ...
                    - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;
                
                
                net.layers{l}.D1 = mexNormalize( max( net.layers{l}.D1 + net.layers{l}.D1Momentum,0) );
                net.layers{l}.D2 = mexNormalize( max(net.layers{l}.D2 + net.layers{l}.D2Momentum,0) );
                continue ;
            end
            
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
            
            net.layers{l}.filtersMomentum = ...
                opts.momentum * net.layers{l}.filtersMomentum ...
                - (lr * net.layers{l}.filtersLearningRate) * ...
                (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
                - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;
            
            net.layers{l}.biasesMomentum = ...
                opts.momentum * net.layers{l}.biasesMomentum ...
                - (lr * net.layers{l}.biasesLearningRate) * ....
                (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
                - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l).dzdw{2} ;
            
            net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
            net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = numel(batch)/batch_time ;
        %     info.train = updateError(opts, info.train, net, res, batch_time) ;
        
        info.train.objective(end) = info.train.objective(end) + sum(double(gather(res(end).x))) ;
        info.train.speed(end) = info.train.speed(end) + speed ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        n = t + numel(batch) - 1 ;
        fprintf(' obj %.1f', ...
            info.train.objective(end)/n) ;
        fprintf('\n') ;
        
        % DISPLAY FOR DEBUG
        sum(double(gather(res(end).x)))
        res = vl_simplenn(net, im, [], res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        sum(double(gather(res(end).x)))
        
        net0.layers{end}.Ymix = im_mix;
        net0.layers{end}.Y1 = im1;
        net0.layers{end}.Y2 = im2;
        
        res0 = vl_simplenn(net0, im, [], res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        sum(double(gather(res0(end).x)))
        
        
        % debug info
        if opts.plotDiagnostics
            figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
        end
    end % next batch
    

end


