
    NFFT = data1.NFFT;
    fs = data1.fs;
    hop = data1.hop;
    N_test = size(idf,1);
    SDR = 0;
    SIR = 0;
    
    for i = 1:N_test

        x1 = test_female{idf(i,1)}{idf(i,2)}{idf(i,3)}.x;

        x2 = test_male{idm(i,1)}{idm(i,2)}{idm(i,3)}.x;
        
        T = min(length(x1),length(x2));

        x1 = x1(1:T);
        x2 = x2(1:T);

        %x1 = x1/norm(x1);
        %x2 = x2/norm(x2);

        mix = (x1+x2);

        X = compute_spectrum(mix,NFFT,hop);
        
%         if param.renorm
%             Xn = renorm_spect_data(abs(X), stds);
%         end
        %epsilon = 1;
        Xn = softNormalize(abs(X),param.epsilon);
        
        % compute decomposition
        param.nu = 0;
        [Z1dm, Z1gn1dm, Z2dm, Zgn2dm] = twolevellasso_gpu_demix(Xn, D1i, Dgn1, D2i, Dgn2, param);
        
        W1H1 = D1*Z1dm;
        W2H2 = D2*Z2dm;

        eps_1 = 1e-6;%eps_1=0;
        V_ap = W1H1.^2 +W2H2.^2 + eps_1;

        % wiener filter

        SPEECH1 = ((W1H1.^2)./V_ap).*X;
        SPEECH2 = ((W2H2.^2)./V_ap).*X;
        speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

        Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
        
%         NSDR = SDR+mean(Parms.NSDR)/N_test;
%         SIR = SIR+mean(Parms.SIR)/N_test;
        
        Parms
        AA{end}.res{i}
%         output{i} = Parms;
        

    end
%     
%     for i=1:length(output)
%         R(i) = mean(output{i}.NSDR);
%     end