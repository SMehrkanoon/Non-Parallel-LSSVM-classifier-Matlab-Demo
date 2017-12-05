
function [alpha_lssvm,b_lssvm,gamma,sig,minBB1]=normal_lssvm_classifier(DATA_ls,DATA_2,type,numFolds)

Xtr=DATA_ls{1};
Xval=DATA_ls{2};
Ytr=DATA_ls{3};
Yval=DATA_ls{4};

X=DATA_2{1};
Y=DATA_2{2};


%%
if strcmp(type,'RBF_kernel')
    
    
    gamma_range = logspace(-2,5,10);
    sigma_range = logspace(-2,4,10);
    
    
    for gamma_idx=1:size(gamma_range,2)
        gamma = gamma_range(gamma_idx);
        
        for sig_idx=1:size(sigma_range,2)
            sig = sigma_range(sig_idx);
            
            
            for k=1:numFolds
                
                K=KernelMatrix(Xtr{k},type, sig);
                m=size(K,1);
                D=diag(Ytr{k});
                omega=D*K*D;
                A= [omega + (1/gamma) * eye(m), Ytr{k};...
                    Ytr{k}' ,0];
                B1= [ones(m,1);0];
                result1=A\B1;
                alpha1=result1(1:m);
                b1=result1(end);
                K2=KernelMatrix(Xtr{k},type, sig, Xval{k});
                yhatval = sign(K2'*D* alpha1  + b1);
                etest1=abs(Yval{k}-yhatval) ;
                num=size(find(etest1>0),1);
                folderror(k)=num/size(Xval{k},1);   % k is for the k-fold
               
                
            end
            BB1(gamma_idx,sig_idx)=mean(folderror);
            
        end
        
    end
 
    
    [minBB1 idx] = min(BB1(:));
    sprintf(' min error = %f\n',minBB1)
    [p q] = ind2sub(size(BB1),idx);
    gamma=gamma_range(p);
    sig=sigma_range(q);
        
    K=KernelMatrix(X,type, sig);
    m=size(K,1);
    D=diag(Y);
    omega=D*K*D;
    
    A= [omega + (1/gamma) * eye(m), Y;...
        Y' ,0];
    B1= [ones(m,1);0];
    result1= A\B1;
    alpha_lssvm=result1(1:m);
    b_lssvm=result1(end);
    
 %%   
elseif strcmp(type,'lin_kernel')    %  for lin_kernel
    
    gamma_range = logspace(0,5,30);
    
    for gamma_idx=1:size(gamma_range,2)
        gamma = gamma_range(gamma_idx);
       
        for k=1:numFolds
            
            K=KernelMatrix(Xtr{k},type);
            m=size(K,1);
            D=diag(Ytr{k});
            omega=D*K*D;
            A= [omega + (1/gamma) * eye(m), Ytr{k};...
                Ytr{k}' ,0];
            B1= [ones(m,1);0];
            result1=A\B1;
            alpha1=result1(1:m);
            b1=result1(end);
            K2=KernelMatrix(Xtr{k},type,[], Xval{k});
            yhatval = sign(K2'*D* alpha1  + b1);
            etest1=abs(Yval{k}-yhatval) ;
            num=size(find(etest1>0),1);
            folderror(k)=num/size(Xval{k},1);   % k is for the k-fold
            
        end
        BB1(gamma_idx)=mean(folderror);
        
    end
    
    [minBB1 idx] = min(BB1(:));
    disp('')
    fprintf(' min CV error = %5.2f\n\n',minBB1)
    [p] = ind2sub(size(BB1),idx);
    gamma=gamma_range(p);
    sig=[];
    
    %% obtaining final model parameters using tuned hyper-parameters
   
    K=KernelMatrix(X,type);
    m=size(K,1);
    D=diag(Y);
    omega=D*K*D;
    
    A= [omega + (1/gamma) * eye(m), Y;...
        Y' ,0];
    B1= [ones(m,1);0];
    result1= A\B1;
    alpha_lssvm=result1(1:m);
    b_lssvm=result1(end);
    
    
end

end


