function [e] = check_eigs (A, B, C, D)

e = qz(D + B*inv(A)*B', C);
