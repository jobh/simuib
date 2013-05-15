
N1 = 140
N2 = 63 


M=20; 
for i=1:M
    I1 = eye(N1, N1);
    I2 = eye(N2, N2);

    A = rand(N1, N1);
    A = A*A'; 
    B = rand(N1, N2);
    C = rand(N2, N2); 
    % C is negative! 
    C = -C*C'; 
     



    AA = [A, B; B', C ];  

    SC = C - B'*inv(A)*B; 
    SA = A - B*inv(C)*B'; 

    BB1 = [inv(A), 0*B; 0*B', inv(SC)] * [I1, 0*B; -B'*inv(A), I2];   
   
    BB1AA = BB1*AA; 

    e = eig(BB1AA);
    e1 = sort(abs(e));
    c1 = e1(1)/e1(size(e1)(1))


    BB2 = [inv(A), 0*B; 0*B', inv(-C)] * [I1, 0*B; -B'*inv(A), I2];   
   
    BB2AA = BB2*AA; 

    e = eig(BB2AA);
    e2 = sort(abs(e));
    c2 = e2(1)/e2(size(e2)(1))

    BB3 = [inv(SA), 0*B; 0*B', inv(-C)] * [I1, 0*B; -B'*inv(SA), I2];   
   
    BB3AA = BB3*AA; 

    e = eig(BB3AA);
    e3 = sort(abs(e));
    c3 = e3(1)/e3(size(e3)(1))

    BB4 = [inv(A), 0*B; 0*B', inv(SC)]  * [I1, -B*inv(SC); 0*B', I2];   
   
    BB4AA = BB4*AA; 

    e = eig(BB4AA);
    e4 = sort(abs(e));
    c3 = e4(1)/e4(size(e4)(1))






%    hold on; 
%    plot(sort(real(e1)));

    


endfor

