function [Parms,wavoutE,wavoutA, E_mag, A_mag,S] = rpca_mask_fun_new(wavinmix,parm,wavinA,wavinE)
    %% parameters

    lambda = parm.lambda;
    nFFT = parm.nFFT;
    winsize = parm.windowsize;
    masktype = parm.masktype;
    gain = parm.gain;
    power = parm.power;
    Fs= parm.fs;
    outputname = parm.outname;

    hop = winsize/4;
    scf = 2/3;
    S = scf * stft(wavinmix, nFFT ,winsize, hop);

   %% use inexact_alm_rpca to run RPCA
    try                
        [A_mag E_mag] = inexact_alm_rpca(abs(S).^power',lambda/sqrt(max(size(S))));
        PHASE = angle(S');            
    catch err
        [A_mag E_mag] = inexact_alm_rpca(abs(S).^power,lambda/sqrt(max(size(S))));
        PHASE = angle(S);
    end
    
    
    %% RNMF
%     K = 100;
%     [F,N] = size(S);
%     
%     %% Init
%     W_ini = abs(randn(F,K)) + 1;
%     H_ini = abs(randn(K,N)) + 1;
%     E_ini = abs(randn(F,N)) + 1;
%     
%     sigma = 0.5;
%     lambda_ast = sqrt(2)*sqrt(size(S,2))*sigma;
%     lambda = 2*sqrt(2)*sigma;
%     
%     tol = 1e-5;
%     n_iter_max = 1000;
%     beta = 1;
%     power = 2;
%     
%     [W, H, E, obj, fit, V_ap] = rnmf(abs(S).^power, beta, n_iter_max, tol, W_ini, H_ini, E_ini, lambda_ast,lambda);
%     
%     
%     A_mag = sqrt((W*H))';
%     E_mag = sqrt(E');
%     PHASE = angle(S');
    
    A = A_mag.*exp(1i.*PHASE);
    E = E_mag.*exp(1i.*PHASE);

    %% binary mask, no mask
    switch masktype                         
      case 1 % binary mask + median filter
        m= double(abs(E)> (gain*abs(A)));                  
        try  
            Emask =m.*S;
            Amask= S-Emask;
        catch err
            Emask =m.*S';
            Amask= S'-Emask;
        end        
      case 2 % no mask
        Emask=E;
        Amask=A;
      otherwise 
          fprintf('masktype error\n');
    end

    %% do istft
    try 
        wavoutE = istft(Emask', nFFT ,winsize, hop)';   
        wavoutA = istft(Amask', nFFT ,winsize, hop)';
    catch err
        wavoutE = istft(Emask, nFFT ,winsize, hop)';   
        wavoutA = istft(Amask, nFFT ,winsize, hop)';
    end

    if ~isempty(outputname)
        wavoutE=wavoutE/max(abs(wavoutE));
        wavwrite(wavoutE,Fs,[outputname,'_E']);

        wavoutA=wavoutA/max(abs(wavoutA));
        wavwrite(wavoutA,Fs,[outputname,'_A']);
    end
    
    %% evaluate
    if exist('wavinA','var')
        if length(wavoutA)==length(wavinA)
            
            sep = [wavoutA , wavoutE]';
            orig = [wavinA , wavinE]';
            
            for i = 1:size( sep, 1)
                [e1,e2,e3] = bss_decomp_gain( sep(i,:), i, orig);
                [sdr(i),sir(i),sar(i)] = bss_crit( e1, e2, e3);
            end
        else
            minlength=min( length(wavoutE), length(wavinE) );
            
            sep = [wavoutA(1:minlength) , wavoutE(1:minlength)]';
            orig = [wavinA(1:minlength) , wavinE(1:minlength)]';
            
            for i = 1:size( sep, 1)
                [e1,e2,e3] = bss_decomp_gain( sep(i,:), i, orig);
                [sdr(i),sir(i),sar(i)] = bss_crit( e1, e2, e3);
            end
        end
        
        Parms.SDR=sdr(2);
        Parms.SIR=sir(2);
        Parms.SAR=sar(2);
    else
        
        Parms.SDR=-1;
        Parms.SIR=-1;
        Parms.SAR=-1;
    end
    
    E_mag = E_mag';
    A_mag = A_mag';