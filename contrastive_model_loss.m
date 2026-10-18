function [loss, gradients, state] = contrastive_model_loss(net, X, tau)
%CONTRASTIVE_MODEL_LOSS Forward pass + NT-Xent loss + gradients for one
%minibatch, for use with dlfeval in a custom training loop.
%   X: D x 2N dlarray, columns 1..N and (N+1)..2N are the two augmented
%   views of the same N samples (paired by index i <-> i+N).
[Z, state] = forward(net, X);
Zn = Z ./ sqrt(sum(Z.^2, 1) + 1e-8); % L2-normalize projection head output
loss = ntxent_loss(Zn, tau);
gradients = dlgradient(loss, net.Learnables);
end
