

 if 1
        
        save_folder = sprintf('%s-s%d-s%d-%s/',model_name,id_1,id_2,date());
        %save_folder = sprintf('../../public_html/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());
        
        try
            unix(sprintf('mkdir %s',save_folder));
            unix(sprintf('chmod 777 %s ',save_folder));
        catch
        end
        
 end
    
     save_file = sprintf('%sresults.mat',save_folder,'s');
    save(save_file,'output','D1','D2','param','fv','r')
    unix(sprintf('chmod 777 %s ',save_file));
    AA{ii,jj}.res = output;