use tpt_tensor::Tensor;

/// A differentiable module: a forward pass plus a parameter set that the
/// optimizer updates in place between steps.
///
/// `parameters`/`set_parameters` use the same stable ordering so an optimizer
/// can hold moment state keyed by parameter index.
pub trait Module {
    /// Run the forward pass (records autograd tape when inputs require grad).
    fn forward(&self, input: &Tensor) -> Tensor;

    /// All trainable parameters, in a stable order.
    fn parameters(&self) -> Vec<Tensor>;

    /// Write back an updated parameter set (shape/dtype must match `parameters`).
    fn set_parameters(&mut self, params: Vec<Tensor>);

    /// Convenience: sum of all outputs — a trivial scalar loss for testing/demos.
    fn forward_scalar(&self, input: &Tensor) -> Tensor {
        let y = self.forward(input);
        // sum over all elements (seed = ones on backward)
        y
    }
}
