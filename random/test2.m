
N1 = 140
N2 = 100 

alpha=0.0001
beta=0.05
gamma=100.0

M=20; 
for i=1:M
    I1 = eye(N1, N1);
    I2 = eye(N2, N2);

    A = rand(N1, N1);
    A1 = gamma*A*A'; 
    A = rand(N1, N1);
    A2 = A*A'; 

    B = rand(N1, N2);
    C = rand(N2, N2); 
    % C is negative! 
    C = -alpha*C*C'; 
     

    AA = [A1 + A2, B; B', C ];  

    SC1 = - C + beta*B'*inv(A1)*B; 
    SC2 = - C + beta*B'*inv(A2)*B; 

    BB1 = [inv(A1 + A2), 0*B; 0*B', inv(SC1) + inv(SC2) ];    
   
    BB1AA = BB1*AA; 

    e = eig(BB1AA);
    e1 = sort(abs(e));
    c1 = e1(1)/e1(size(e1)(1))

    SC1 =  +  beta*B'*inv(A1)*B; 
    SC2 =  + beta*B'*inv(A2)*B; 
    SC3 = - C ; 

    BB1 = [inv(A1 + A2), 0*B; 0*B', inv(SC1) + inv(SC2) + inv(SC3) ];    
   
    BB1AA = BB1*AA; 

    e = eig(BB1AA);
    e1 = sort(abs(e));
    c2 = e1(1)/e1(size(e1)(1))






%    hold on; 
%    plot(sort(real(e1)));

    


endfor

