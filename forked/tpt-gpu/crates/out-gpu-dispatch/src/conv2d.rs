// Conv2d dispatch — 2-D convolution (NCHW layout).

/// Dispatch conv2d — NCHW × OIHW → NCHW
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    x: &[f32],
    weight: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    #[cfg(feature = "hardware")]
    {
        // Hardware path placeholder — falls through to fallback.
    }
    fallback_conv2d(
        x, weight, n, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fallback_conv2d(
    x: &[f32],
    weight: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;

    let mut out = vec![0.0f32; n * c_out * h_out * w_out];

    for batch in 0..n {
        for g in 0..groups {
            for oc in 0..c_out_per_g {
                let out_ch = g * c_out_per_g + oc;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f32;
                        for ic in 0..c_in_per_g {
                            let in_ch = g * c_in_per_g + ic;
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * stride + fh * dilation;
                                    let iw = ow * stride + fw * dilation;
                                    let x_val = if ih >= padding
                                        && iw >= padding
                                        && ih < h + padding
                                        && iw < w + padding
                                    {
                                        let rh = ih - padding;
                                        let rw = iw - padding;
                                        x[((batch * c_in + in_ch) * h + rh) * w + rw]
                                    } else {
                                        0.0
                                    };
                                    let w_val =
                                        weight[((out_ch * c_in_per_g + ic) * kh + fh) * kw + fw];
                                    acc += x_val * w_val;
                                }
                            }
                        }
                        out[((batch * c_out + out_ch) * h_out + oh) * w_out + ow] = acc;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_filter() {
        // 1×1 conv with identity kernel → output == input
        let x = vec![1.0_f32, 2.0, 3.0, 4.0]; // N=1, C=1, H=2, W=2
        let weight = vec![1.0_f32]; // C_out=1, C_in=1, kH=1, kW=1
        let out = fallback_conv2d(&x, &weight, 1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sum_filter() {
        // 1×2 conv sums adjacent cols
        let x = vec![1.0_f32, 2.0, 3.0, 4.0]; // N=1, C=1, H=2, W=2
        let weight = vec![1.0_f32, 1.0]; // C_out=1, C_in=1, kH=1, kW=2
        let out = fallback_conv2d(&x, &weight, 1, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1);
        assert_eq!(out.len(), 2);
        // row 0: [1,2] → 3; row 1: [3,4] → 7
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_shape() {
        // N=2, C_in=3, H=4, W=4 → C_out=8, kH=3, kW=3, stride=1, pad=1 → H_out=4, W_out=4
        let x = vec![0.0f32; 2 * 3 * 4 * 4];
        let weight = vec![0.0f32; 8 * 3 * 3 * 3];
        let out = fallback_conv2d(&x, &weight, 2, 3, 4, 4, 8, 3, 3, 1, 1, 1, 1);
        assert_eq!(out.len(), 2 * 8 * 4 * 4);
    }
}
