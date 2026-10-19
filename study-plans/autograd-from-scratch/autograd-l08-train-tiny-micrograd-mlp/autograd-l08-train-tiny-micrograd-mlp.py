import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:
    """
    Returns final predictions, final loss, trained weights,
    trained biases, and pre-update loss history.
    """
    # Clone parameters to preserve the original tensors.
    trained_weights = [w.clone() for w in weights]
    trained_biases = [b.clone() for b in biases]

    loss_history = []

    for _ in range(steps):
        # Forward pass: retain activations for backpropagation.
        activations = [inputs]
        h = inputs

        for w, b in zip(trained_weights, trained_biases):
            h = torch.tanh(h @ w.T + b)
            activations.append(h)

        predictions = h.squeeze(-1)
        loss = ((predictions - targets) ** 2).sum()

        # Record loss before updating parameters.
        loss_history.append(loss.clone())

        # Backward pass: calculate gradients manually.
        # dL/dp for the summed squared error.
        delta = 2 * (predictions - targets).unsqueeze(-1)

        # Derivative of the final tanh activation.
        delta = delta * (1 - activations[-1] ** 2)

        weight_grads = [None] * len(trained_weights)
        bias_grads = [None] * len(trained_biases)

        for layer in range(len(trained_weights) - 1, -1, -1):
            # Gradients for this layer's weights and biases.
            weight_grads[layer] = delta.T @ activations[layer]
            bias_grads[layer] = delta.sum(dim=0)

            # Propagate the gradient to the previous layer.
            if layer > 0:
                delta = delta @ trained_weights[layer]
                delta = delta * (1 - activations[layer] ** 2)

        # Simultaneous gradient-descent update.
        trained_weights = [
            w - learning_rate * gw
            for w, gw in zip(trained_weights, weight_grads)
        ]
        trained_biases = [
            b - learning_rate * gb
            for b, gb in zip(trained_biases, bias_grads)
        ]

    # Compute results using the final parameters.
    h = inputs
    for w, b in zip(trained_weights, trained_biases):
        h = torch.tanh(h @ w.T + b)

    predictions = h.squeeze(-1)
    final_loss = ((predictions - targets) ** 2).sum()

    return (
        predictions,
        final_loss,
        trained_weights,
        trained_biases,
        loss_history
    )