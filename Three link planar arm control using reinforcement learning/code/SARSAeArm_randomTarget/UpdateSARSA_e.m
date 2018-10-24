function [Q, e] = UpdateSARSA_e( s, a, r, sp, ap, tab, etrace, alpha, gamma, lambda )



Q = tab;
e = etrace;

delta  =  r + gamma * Q(sp,ap) - Q(s,a);
e(s,a) =  e(s,a) + 1;

Q = Q + alpha * delta * e;
e = gamma * lambda * e;


