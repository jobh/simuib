
N1 = 240;
N2 = 140; 

eps= 1.0e-1; 

M=20; 
for i=1:M

    I1 = eye(N1,N1); 
    A = rand(N1, N1);
    A = A*A'; 
    B = rand(N1, N2);
    C = rand(N2, N2); 
    % C is negative! 
    C1 = -C*C'; 
    C = rand(N2, N2); 
    C2 = -C*C'; 
     



    AA = [A + 1/eps*B*inv(-C1)*B', B; B', C1 + C2 ];  

%    BB1 = [inv(A + B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A)*B) + inv(-C1 + -C2 + B'*inv(inv(B*-C1*B' + eps*I1))*B)  ];    
    BB1 = [inv(A + 1/eps*B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A)*B) + inv(-C1 + -C2 + B'*inv(1/eps*B*-inv(C1)*B')*B)  ];    
    BB1 = [inv(A + 1/eps*B*inv(-C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A)*B) + inv(-C1 + -C2 + eps*-C1)  ];    
%    BB1 = [inv(A + B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(inv(B*-C1*B'))*B)  ]    
%    BB1 = [inv(A + B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A)*B)  ];    
   
    BB1AA = BB1*AA; 

    e = eig(BB1AA);
    e1 = sort(abs(e));
    c1 = e1(1)/e1(size(e1)(1))

    BB2 = [inv(A + 1/eps*B*inv(-C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A + 1/eps*B*inv(-C1)*B')*B)];    
%    BB1 = [inv(A + B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(inv(B*-C1*B'))*B)  ]    
%    BB1 = [inv(A + B*inv(C1)*B'), 0*B; 0*B', inv(-C1 + -C2 + B'*inv(A)*B)  ];    
   
    BB2AA = BB2*AA; 

    e = eig(BB2AA);
    e2 = sort(abs(e));
    c2 = e2(1)/e2(size(e2)(1))




%    hold on; 
%    plot(sort(real(e1)));

    


endfor

