function resout = vl_ensemble(net, res, testing)

resout = res;


	%combine the outputs from the branches and renormalize
	m = size(res,2);

    aux = res{1}(end).x;
    for i=2:m-1
        aux = aux.*res{i}(end).x;
    end
    aux2 = 0*aux;
    aux2(:,:,1:2:end,:) = aux(:,:,1:2:end,:)+aux(:,:,2:2:end,:);
    aux2(:,:,2:2:end,:) = aux2(:,:,1:2:end,:);
    
    resout{end}(1).x = aux./(eps+aux2);

	if ~testing
	%fprop
      	resout{end}(end+1).x = vl_fit(resout{end}(1).x,net.layers{end}.Ymix,net.layers{end}.Y1,net.layers{end}.Y2,[],'loss',net.layers{end}.loss);

	 %backprop
      	resout{end}(end).dzdx = vl_fit(resout{end}(1).x,net.layers{end}.Ymix,net.layers{end}.Y1,net.layers{end}.Y2,1,'loss',net.layers{end}.loss);

     %compute gradients for branches
	
	tmp = resout{end}(end).dzdx;
	den = aux2(:,:,1:2:end,:).^2;
	resout{end}(end).dzdx(:,:,1:2:end,:) = tmp(:,:,1:2:end,:).*(aux(:,:,2:2:end,:)./den); 
	resout{end}(end).dzdx(:,:,2:2:end,:) = tmp(:,:,2:2:end,:).*(aux(:,:,1:2:end,:)./den); 
	for i=1:m-1
		resout{i}(end+1).dzdx = resout{end}(end).dzdx .* (aux ./ (eps+res{i}(end).x)) ;
	end

	end


