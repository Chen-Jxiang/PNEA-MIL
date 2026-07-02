import numpy as np
import torch
import torch.nn as nn


class PNEA_MIL(torch.nn.Module):
    """Positive-Negative Evidence Analysis MIL model.

    This implementation follows the PNEA-MIL formulation described in the
    manuscript. In the binary case used in the paper, each slide is represented
    by two groups of evidence axes: evidence for the negative class and evidence
    for the positive class. The model can also be generalized to C > 2 classes;
    in that case, each class has its own set of D non-negative evidence axes.

    Parameters
    ----------
    D_in : int
        Dimension of the patch-level feature vector. The manuscript uses
        1024-dimensional UNI features.
    D : int
        Number of evidence axes per class. The manuscript uses D = 64 unless
        otherwise specified.
    C : int
        Number of classes. The manuscript presents the binary case, C = 2.
    seed : int
        Random seed for reproducibility.
    lr : float
        Learning rate for Adam. The manuscript uses 1e-4.
    weight_decay : float
        Optional L2 weight decay used by Adam. The default is 0, so the main
        regularization is the explicit L1 penalty on the non-negative evidence
        weights.
    lambda_l1 : float
        Strength of the L1 penalty on the non-negative evidence weights.
        The manuscript uses lambda = 5 unless otherwise specified.
    class_weight : torch.Tensor or None
        Optional class weights for cross-entropy loss. This is useful when the
        class distribution is imbalanced.
    """

    def __init__(
        self,
        D_in: int = 1024,
        D: int = 64,
        C: int = 2,
        seed: int = 0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lambda_l1: float = 5.0,
        class_weight: torch.Tensor | None = None,
    ):
        super(PNEA_MIL, self).__init__()

        # Fix random seeds so that model initialization is reproducible.
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.D = D
        self.C = C
        self.lambda_l1 = lambda_l1

        # For C = 2, cross-entropy over two logits is equivalent to binary
        # logistic regression based on the logit difference. Specifically,
        # softmax([E-, E+]) gives sigmoid(E+ - E-) for the positive class.
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weight)

        # Patch-level evidence mapping. In the manuscript notation, this linear
        # layer contains the class-specific mappings f^+ and f^- in the binary
        # case. More generally, it maps each patch feature x_{n,p} to C x D raw
        # evidence scores, one D-dimensional evidence vector per class.
        self.f = nn.Linear(D_in, C * D)

        # Class-specific evidence weights. ReLU is applied to this parameter in
        # forward(), so the effective weights are non-negative. This preserves
        # the interpretation that each evidence axis can only increase the
        # evidence score for its corresponding class.
        self.W = nn.Parameter(
            0.1 * torch.abs(torch.rand(C, D)),
            requires_grad=True,
        )

        # Bias term for each class logit. The binary manuscript equation omits
        # this term for simplicity; with C = 2, the implemented probability is
        # sigmoid((E+ + b+) - (E- + b-)).
        self.b = nn.Parameter(torch.zeros(C), requires_grad=True)

        # Optimizer. Bias parameters are excluded from weight decay. The explicit
        # L1 penalty on ReLU(W) is added in return_loss().
        decay_params = [self.W]
        no_decay_params = [self.b]

        for name, parameter in self.f.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("bias"):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        self.optim = torch.optim.Adam(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def forward(self, X: torch.Tensor, return_evidence: bool = False):
        """Compute slide-level logits from a bag of patch features.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with shape [N, P, D_in], where N is the number of
            slides in the batch, P is the number of patches per slide, and D_in
            is the patch-feature dimension.
        return_evidence : bool
            If True, also return patch-level and slide-level evidence tensors.
            These tensors are useful for evidence visualization and post-hoc
            interpretation, as described in the manuscript.

        Returns
        -------
        logits : torch.Tensor
            Slide-level class logits with shape [N, C].
        evidence_npck : torch.Tensor, optional
            Patch-level non-negative evidence values with shape [N, P, C, D].
        evidence_nck : torch.Tensor, optional
            Slide-level max-pooled evidence values with shape [N, C, D].
        """
        N, P, D_in = X.shape

        # Apply the evidence mapping to every patch independently.
        X_nd = X.reshape(N * P, D_in)
        raw_evidence = self.f(X_nd)

        # Convert raw diagnostic signals into non-negative evidence magnitudes.
        # This corresponds to t_{n,p} = ReLU(f(x_{n,p})) in the manuscript.
        evidence_npck = torch.relu(raw_evidence).view(N, P, self.C, self.D)

        # Max pooling over patches selects the strongest expression of each
        # evidence axis across the slide, matching the MIL assumption that a
        # localized informative region can drive the slide-level label.
        evidence_nck = evidence_npck.max(dim=1).values

        # Combine slide-level evidence axes using non-negative weights. For the
        # binary case, logits[:, 0] can be interpreted as negative-class evidence
        # and logits[:, 1] as positive-class evidence.
        nonnegative_W = torch.relu(self.W)
        logits = torch.einsum("nck,ck->nc", evidence_nck, nonnegative_W) + self.b

        if return_evidence:
            return logits, evidence_npck, evidence_nck
        return logits

    def return_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the PNEA-MIL training loss.

        The loss combines cross-entropy classification loss with an L1 penalty
        on the non-negative evidence weights:

            L = CE(logits, y) + lambda_l1 * ||ReLU(W)||_1.

        In the binary case, this corresponds to the manuscript's BCE-based
        formulation because the two-logit cross-entropy is equivalent to binary
        cross-entropy on the positive-negative logit difference.
        """
        logits = self.forward(X)
        classification_loss = self.classification_loss(logits, y)

        # Penalize the effective non-negative weights, not the unconstrained raw
        # parameters. This encourages the model to use a smaller subset of
        # evidence axes, as described in the manuscript.
        l1_loss = self.lambda_l1 * torch.relu(self.W).mean()

        return classification_loss + l1_loss
