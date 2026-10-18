function [loss, grad] = test_ntxent(Z, tau)
Zn = Z ./ sqrt(sum(Z.^2, 1));
loss = ntxent_loss(Zn, tau);
grad = dlgradient(loss, Z);
end
