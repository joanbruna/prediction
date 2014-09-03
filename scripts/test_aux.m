

cc = bwconncomp(SA>0.01);

for i=1:length(cc.PixelIdxList)
    L(i) = length(cc.PixelIdxList{i});
end

[m,id] = max(L);

R = {};
B = {};
count = 0;
A_rec = 0*A;
A_out = A_rec;
for i=1:length(cc.PixelIdxList)
    
    if length(cc.PixelIdxList{i}) > 20
        count = count+1;
        
        
    B{count} = 0*A;
    
    %[j,k] = ind2sub([100,304],cc.PixelIdxList{i}(round(end/2)));
    
    B{count}(cc.PixelIdxList{i}) = A(cc.PixelIdxList{i});   
   
    R{count} = D*B{count};
    
    A_rec(cc.PixelIdxList{i}) = A(cc.PixelIdxList{i});
    else
        A_out(cc.PixelIdxList{i}) = A(cc.PixelIdxList{i});
    end
end


y = wienerFilter2(R,Smix);

for i=1:count
    
    audiowrite(['../../public_html/wav/x' num2str(i) '.wav'],y{i},16000)
    
end


%%

% V_rec = (D*A_rec).*(repmat(sqrt(norms.^2+epsilon^2),size(Pmix,1),1));
% y_rec= invert_spectrum(abs(V_rec).*exp(i*angle(Smix)),NFFT , hop);

R_rec{1} = D*A_rec;
R_rec{2} = D*A_out;
y_rec = wienerFilter2(R_rec,Smix);



%%

G = [3,13; 17,24; 25,37; 42,50; 51,60; 68,82; 83,86; 87, 96];


B = {};
for h=1:size(G,1)
B{h} =0*A;
end

A_out = 0*A;
for i=1:length(cc.PixelIdxList)
    
    if length(cc.PixelIdxList{i}) > 20
        
        [j,k] = ind2sub([100,304],cc.PixelIdxList{i}(round(end/2)));
        
        for h=1:size(G,1)
        
        if j >= G(h,1) && j <= G(h,2)
        
        B{h}(cc.PixelIdxList{i}) = A(cc.PixelIdxList{i});
       
        end
        
        end
        
    else
        A_out(cc.PixelIdxList{i}) = A(cc.PixelIdxList{i});
    end
end

R_group = {};
for h=1:size(G,1)
R_group{h} =D*B{h};
end
R_group{h+1} =D*A_out;

y_group = wienerFilter2(R_group,Smix);

for i=1:length(y_group)
    
    audiowrite(['../../public_html/wav2/reg' num2str(i) '.wav'],y_group{i},16000)
    
end

