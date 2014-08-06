

function obj = evaluate_obj(V,W,Wn,H,Hn,lambda1)


N = size(V,2);
K = size(W,2);
Kn = size(Wn,2);


V_ap = W*H+Wn*Hn;

obj = 0.5*norm(V-V_ap,'fro')^2 + lambda1*sum(H(:));