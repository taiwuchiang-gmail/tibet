function [loss, gradients, state] = contrastive_model_loss(net, X, tau, labels)
%CONTRASTIVE_MODEL_LOSS Forward pass + contrastive loss + gradients for
%one minibatch, for use with dlfeval in a custom training loop.
%   X: dlarray with 2N observations; observations 1..N and (N+1)..2N are
%   the two augmented views of the same N samples (paired i <-> i+N).
%   LABELS (optional): 1 x N known class per sample, -1 if unknown. When
%   given, uses supcon_loss; otherwise ntxent_loss.
[Z, state] = forward(net, X);
Zn = Z ./ sqrt(sum(Z.^2, 1) + 1e-8); % L2-normalize projection head output
if nargin < 4 || isempty(labels)
    loss = ntxent_loss(Zn, tau);
else
    loss = supcon_loss(Zn, tau, labels);
end
gradients = dlgradient(loss, net.Learnables);
end
