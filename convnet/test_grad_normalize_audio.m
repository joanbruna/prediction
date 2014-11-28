% first make some data
n = 20;
m = 3;
r = 12;



Y = 100*rand(1,r,n,m);
X = 100*rand(1,r,n,m);


param = [0 1e-5 0 0];

M = vl_nnnormalize_audio(X,param,[]);

obj = 0.5*sum( (Y(:) - M(:).*Y(:)).^2);

dzdy = (M.*Y-Y).*Y;

dzdx = vl_nnnormalize_audio(X,param,dzdy);


% dH1
eps_1 = 1e-7;
dX = eps_1*randn(size(X));
X_ = X + dX;

M_ = vl_nnnormalize_audio(X_,param);
obj_ = 0.5*sum( (Y(:) - M_(:).*Y(:)).^2);

[obj_-obj, dzdx(:)'*dX(:)]/eps_1
