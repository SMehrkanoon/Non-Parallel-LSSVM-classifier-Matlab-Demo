
function [beta1_final,beta2_final,b1_final,b2_final,sig,gamma,minBB1]=non_parallel_lssvm_classifier(DATA,Xtr,DATA_val,type,ratio,numFolds,par_lssvm)

X1=DATA{1};
X2=DATA{2};
X1tr=Xtr{1};
X2tr=Xtr{2};
Xval=DATA_val{1};
Yval=DATA_val{2};

if strcmp(type,'RBF_kernel')
    
    gamma=1*par_lssvm(1);
    sig=par_lssvm(2);
 
    for k=1:numFolds
        % ========= opt 1 =============
        
        n1=size(X1tr{k},1);
        n2=size(X2tr{k},1);
        
        K11=KernelMatrix(X1tr{k},type, sig);
        Kbar=KernelMatrix(X1tr{k},type, sig,X2tr{k});
        K22=KernelMatrix(X2tr{k},type, sig);

        H1=K11 + (1/gamma/ratio)*eye(size(K11,1));
        H2=Kbar;
        H3=K22  + (1/gamma)*eye(size(K22,1));
        
      A= [H1 , -H2 , ones(n1,1);...
            -H2' , H3, -ones(n2,1);...
            ones(n1,1)',  -ones(n2,1)', 0];

        B= [zeros(n1,1); ones(n2,1); 0];
        result1=A\B;
        beta11=result1(1:n1);
        beta21=result1(n1+1:n1+n2);
        b1=result1(end);
        
  
        % =========== opt 2 ===============
        K11=KernelMatrix(X1tr{k},type, sig);   
        K22=KernelMatrix(X2tr{k},type, sig);   
        H1=K22 + (1/gamma/ratio)*eye(size(K22,1));
        H2=Kbar';
        H3=K11  + (1/gamma)*eye(size(K11,1));
        A= [H1 , H2 , ones(n2,1);...
            H2' , H3, ones(n1,1);...
            ones(n2,1)',  ones(n1,1)', 0];
        B= [zeros(n2,1); ones(n1,1); 0];
        result2=A\B;
        beta12=result2(1:n2);
        beta22=result2(n2+1:n1+n2);
        b2=result2(end);
        

        
        % ==========  Validation ==============
        
        K1val=KernelMatrix(Xval{k},type, sig, X1tr{k});
        K2val=KernelMatrix(Xval{k},type, sig, X2tr{k});
        dis1=abs(K1val* beta11 - K2val*beta21 + b1);%./norm(alpha1,2);
        dis2=abs(K2val* beta12 + K1val*beta22 + b2);%./norm(alpha2,2);
        Dist=[dis1 , dis2];
        [I,C] = min(Dist,[],2);
        C(find(C==2)) = -1;
        val_error=abs(Yval{k}-C);
        num=size(find(val_error>0),1);
        folderror(k)=num/size(Xval{k},1);   % k is for the k-fold
        
       
    end
    
    minBB1=mean(folderror);
    
    % ======= opt 1 ==============
    
    n1=size(X1,1);
    n2=size(X2,1);
    K11=KernelMatrix(X1,type, sig);
    Kbar=KernelMatrix(X1,type, sig,X2);
    K22=KernelMatrix(X2,type, sig);
   
    H1=K11 + (1/gamma/ratio)*eye(size(K11,1));
    H2=Kbar;
    H3=K22  + (1/gamma)*eye(size(K22,1));
    
    A= [H1 , -H2 , ones(n1,1);...
        -H2' , H3, -ones(n2,1);...
        ones(n1,1)',  -ones(n2,1)', 0];
    B= [zeros(n1,1); ones(n2,1); 0];
    
    result1=A\B;
    beta11=result1(1:n1);
    beta21=result1(n1+1:n1+n2);
    b1_final=result1(end);
    beta1_final={beta11,beta21};
   

    % ========= opt 2 =============

    K11=KernelMatrix(X1,type, sig);   
    K22=KernelMatrix(X2,type, sig);  
    
    H1=K22 + (1/gamma/ratio)*eye(size(K22,1));
    H2=Kbar';
    H3=K11  + (1/gamma)*eye(size(K11,1));
    
  
    A= [H1 , H2 , ones(n2,1);...
        H2' , H3, ones(n1,1);...
        ones(n2,1)',  ones(n1,1)', 0];
    B= [zeros(n2,1); ones(n1,1); 0];

    result2=A\B;
    beta12=result2(1:n2);
    beta22=result2(n2+1:n1+n2);
    b2_final=result2(end);
    beta2_final={beta12,beta22};
   
    
   
elseif strcmp(type,'lin_kernel')    %  for lin_kernel
    

    gamma=par_lssvm(1);
    
  
    for k=1:numFolds
        %==== opt 1 =========
        
        n1=size(X1tr{k},1);
        n2=size(X2tr{k},1);
        K11=KernelMatrix(X1tr{k},type);
        Kbar=KernelMatrix(X1tr{k},type,[],X2tr{k});
        K22=KernelMatrix(X2tr{k},type);
        H1=K11 + (1/gamma/ratio)*eye(size(K11,1));
        H2=Kbar;
        H3=K22  + (1/gamma)*eye(size(K22,1));
        A= [H1 , -H2 , ones(n1,1);...
            -H2' , H3, -ones(n2,1);...
            ones(n1,1)',  -ones(n2,1)', 0];
        B= [zeros(n1,1); ones(n2,1); 0];
        result1=A\B;
        beta11=result1(1:n1);
        beta21=result1(n1+1:n1+n2);
        b1=result1(end);
        
        
        %======= opt 2 ============

        K11=KernelMatrix(X1tr{k},type);   
        K22=KernelMatrix(X2tr{k},type);   
        H1=K22 + (1/gamma/ratio)*eye(size(K22,1));
        H2=Kbar';
        H3=K11  + (1/gamma)*eye(size(K11,1));
        A= [H1 , H2 , ones(n2,1);...
            H2' , H3, ones(n1,1);...
            ones(n2,1)',  ones(n1,1)', 0];
        B= [zeros(n2,1); ones(n1,1); 0];
        result2=A\B;
        beta12=result2(1:n2);
        beta22=result2(n2+1:n1+n2);
        b2=result2(end);

        % === validation =========
        
        K1val=KernelMatrix(Xval{k},type,[], X1tr{k});
        K2val=KernelMatrix(Xval{k},type,[], X2tr{k});  
        dis1=abs(K1val* beta11 - K2val*beta21 + b1);%./norm(alpha1,2);
        dis2=abs(K2val* beta12 + K1val*beta22 + b2);%./norm(alpha2,2);
        Dist=[dis1 , dis2];
        [I,C] = min(Dist,[],2);
        C(find(C==2)) = -1;
        val_error=abs(Yval{k}-C);
        num=size(find(val_error>0),1);
        folderror(k)=num/size(Xval{k},1);   % k is for the k-fold
        
    end

    minBB1=mean(folderror);
    
    sig=[];
    
   
    
    %=========== opt 1 ========
    
    n1=size(X1,1);
    n2=size(X2,1);
    K11=KernelMatrix(X1,type, sig);
    Kbar=KernelMatrix(X1,type, sig,X2);
    K22=KernelMatrix(X2,type, sig);
    H1=K11 + (1/gamma/ratio)*eye(size(K11,1));
    H2=Kbar;
    H3=K22  + (1/gamma)*eye(size(K22,1));
    A= [H1 , -H2 , ones(n1,1);...
        -H2' , H3, -ones(n2,1);...
        ones(n1,1)',  -ones(n2,1)', 0];
    B= [zeros(n1,1); ones(n2,1); 0];
    result1=A\B;
    beta11=result1(1:n1);
    beta21=result1(n1+1:n1+n2);
    b1_final=result1(end);
    beta1_final={beta11,beta21};
  
    
    
    % ======== opt 2 ==============
     
    K11=KernelMatrix(X1,type, sig);  
    K22=KernelMatrix(X2,type, sig);  
    H1=K22 + (1/gamma/ratio)*eye(size(K22,1));
    H2=Kbar';
    H3=K11  + (1/gamma)*eye(size(K11,1));
    A= [H1 , H2 , ones(n2,1);...
        H2' , H3, ones(n1,1);...
        ones(n2,1)',  ones(n1,1)', 0];
    B= [zeros(n2,1); ones(n1,1); 0];
    result2=A\B;
    beta12=result2(1:n2);
    beta22=result2(n2+1:n1+n2);
    b2_final=result2(end);
    beta2_final={beta12,beta22};
    
   
end

end


