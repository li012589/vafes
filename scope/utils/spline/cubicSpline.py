import torch, math
import torch.nn.functional as F

from .spline import SplineFn


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def coefficient(params):
    '''
    convert neural network params to bin params.
    params contains:
        unnormalized_widths (ndarray of k elements);
        unnormalized_heights (ndarray of k elements);
        unnorm_derivatives_left (ndarray of 1 element, or float);
        unnorm_derivatives_right (ndarray of 1 element, or float);
        min_bin_width (float);
        min_bin_height (float).
    '''
    unnormalized_widths = params[0]
    unnormalized_heights = params[1]
    unnorm_derivatives_left = params[2]
    unnorm_derivatives_right = params[3]
    min_bin_width = params[4]
    min_bin_height = params[5]

    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    cumwidths[..., -1] = 1
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

    cumheights[..., -1] = 1
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    slopes = heights / widths
    min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                torch.abs(slopes[..., 1:]))
    min_something_2 = (
        0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
        / (widths[..., :-1] + widths[..., 1:])
    )
    min_something = torch.min(min_something_1, min_something_2)

    derivatives_left = torch.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., None]
    derivatives_right = torch.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., None]

    derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.sign(slopes[..., 1:]))
    derivatives = torch.cat([derivatives_left,
                             derivatives,
                             derivatives_right], dim=-1)

    return (derivatives, slopes, cumwidths, widths, cumheights, heights)


class GeneralCubicFn(SplineFn):
    '''
    A general functional Cubic Spline of the form
     y_i = A_i * x_i ^3 + B_i * x_i ^2 + C_i * x_i + D_i.
    '''
    def __init__(self, name="CubicSpline"):
        super().__init__(name)

    @staticmethod
    def _forward(x, params, *args, **kwargs):
        return params[0] * x**3 + params[1] * x**2 + params[2] * x + params[3]

    @staticmethod
    def _forwardnGrad(x, params, *args, **kwargs):
        return params[0] * x**3 + params[1] * x**2 + params[2] * x + params[3], 3 * params[0] * x**2 + 2 * params[1] * x + params[2]


    @staticmethod
    def _grad(x, params, *args, **kwargs):
        return 3 * params[0] * x**2 + 2 * params[1] * x + params[2]

    @staticmethod
    def _inverse(y, params, quadraticThreshold=1e-6, solver='original', *args, **kwargs):
        '''
        Args:
            y (ndarray, batch x 1): the outputs from the spline.
            params (list) contains:
                A (ndarray, batch x 1): the 1st coefficient;
                B (ndarray, batch x 1): the 2nd coefficient;
                C (ndarray, batch x 1): the 3rd coefficient;
                D (ndarray, batch x 1): the 4th coefficient;
            quadraticThreshold(float): tolerance for quadratic case.
            solver (str): retained for checkpoint compatibility; only the
                original inverse solver is kept.
        '''
        return GeneralCubicFn._inverse_original(y, params, quadraticThreshold)

    @staticmethod
    def _inverse_original(y, params, quadraticThreshold=1e-6):
        '''
        Original trigonometric solver for cubic inverse.
        Args:
            y (ndarray, batch x 1): the outputs from the spline.
            params (list) contains:
                A (ndarray, batch x 1): the 1st coefficient;
                B (ndarray, batch x 1): the 2nd coefficient;
                C (ndarray, batch x 1): the 3rd coefficient;
                D (ndarray, batch x 1): the 4th coefficient;
            quadraticThreshold(float): tolerance for quadratic case.
        '''
        inputs_a = params[0]
        inputs_b = params[1]
        inputs_c = params[2]
        inputs_d = params[3]

        # Modified coefficients for solving the cubic.
        original_shape = y.shape

        inputs_a = inputs_a.flatten().unsqueeze(-1)
        inputs_b = inputs_b.flatten().unsqueeze(-1)
        inputs_c = inputs_c.flatten().unsqueeze(-1)
        inputs_d = inputs_d.flatten().unsqueeze(-1)
        y_flat = y.flatten().unsqueeze(-1)

        num_elements = inputs_a.shape[0]

        inputs_b_ = (inputs_b / inputs_a) / 3.
        inputs_c_ = (inputs_c / inputs_a) / 3.
        inputs_d_ = (inputs_d - y_flat) / inputs_a

        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0 # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        outputs = torch.zeros(num_elements, 1, device=inputs_a.device, dtype=inputs_a.dtype)

        # Deal with one root cases.
        if one_root_mask.sum() > 0:
            sqrt_neg_disc = torch.sqrt(-discriminant[one_root_mask])
            p = cbrt((-depressed_1[one_root_mask] + sqrt_neg_disc) / 2.)
            q = cbrt((-depressed_1[one_root_mask] - sqrt_neg_disc) / 2.)
            outputs[one_root_mask] = ((p + q) - inputs_b_[one_root_mask])

        if three_roots_mask.sum() > 0:
            theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
            theta /= 3.

            cubic_root_1 = torch.cos(theta)
            cubic_root_2 = torch.sin(theta)

            root_1 = cubic_root_1
            root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
            root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

            root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
            root_shift = (-inputs_b_[three_roots_mask])

            root_1_ = root_1 * root_scale + root_shift
            root_2_ = root_2 * root_scale + root_shift
            root_3_ = root_3 * root_scale + root_shift

            roots_ = torch.stack([root_1_, root_2_, root_3_], dim=-1)

            in_range_mask = (roots_ >= 0) & (roots_ <= 1)

            roots_in_range = torch.where(in_range_mask, roots_, float('inf'))
            selected_roots = roots_in_range.min(dim=-1)[0]

            outputs[three_roots_mask] = selected_roots

        # Deal with a -> 0 (almost quadratic) cases.
        quadratic_mask = (inputs_a.abs() < quadraticThreshold) & (inputs_b.abs() >= quadraticThreshold)
        if quadratic_mask.sum() > 0:
            a = inputs_b[quadratic_mask]
            b = inputs_c[quadratic_mask]
            c = (inputs_d[quadratic_mask] - y_flat[quadratic_mask])
            alpha = (-b + torch.sqrt(torch.clip(b.pow(2) - 4*a*c, min=0))) / (2*a)
            outputs[quadratic_mask] = alpha

        # Deal with a,b -> 0 (almost linear) cases.
        linear_mask = (inputs_a.abs() < quadraticThreshold) & (inputs_b.abs() < quadraticThreshold)
        if linear_mask.sum() > 0:
            b = inputs_c[linear_mask]
            c = (inputs_d[linear_mask] - y_flat[linear_mask])
            alpha = -c / b
            outputs[linear_mask] = alpha

        return outputs.reshape(original_shape)

class CubicBernsteinFn(GeneralCubicFn):
    r'''
    b_{v, n}(x) = C^n_v x^v(1-x)^{n-v}, x \in [0,1];
    '''
    def __init__(self, name="CubicBernsteinFn"):
        super().__init__(name)

    @staticmethod
    def _forward(x, params, *args, **kwargs):
        return params[0] * (1 - x)**3 + params[1] * x * (1 - x)**2 + params[2] * x**2 * (1 - x) + params[3] * x**3

    @staticmethod
    def _forwardnGrad(x, params, *args, **kwargs):
        grad = -3 * params[0] * (1 - x)**2 + params[1] * (1 - x)**2 - 2 * params[1] * x * (1 - x) + 2 * params[2] * x * (1 - x) - params[2] * x**2 + 3 * params[3] * x**2
        results = params[0] * (1 - x)**3 + params[1] * x * (1 - x)**2 + params[2] * x**2 * (1 - x) + params[3] * x**3
        return results, grad

    @staticmethod
    def _grad(x, params, *args, **kwargs):
        return 3 * (-params[0] + params[1] - params[2] + params[3]) * x**2 + 2 * (3 * params[0] - 2 * params[1] + params[2]) * x - 3 * params[0] + params[1]

    @classmethod
    def _inverse(cls, y, params, quadraticThreshold=1e-6, solver='original', *args, **kwargs):
        '''
        convert the params to general cubic function form.
        '''
        a, b, c, d = params
        a_ = b - a - c + d
        b_ = 3 * a - 2 * b + c
        c_ = b - 3 * a
        d_ = a
        return super()._inverse(y, (a_, b_, c_, d_), quadraticThreshold=quadraticThreshold, solver=solver)


class SteffenBernsteinSplineFn(CubicBernsteinFn):
    '''
    Steffen cubic spline in Bernstein form, to eliminate errors at x == 1.
    '''
    def __init__(self, name="SteffenBernsteinFn"):
        super().__init__(name)

    @staticmethod
    def _preprocess(params):
        '''
        params contains:
            unnormalized_widths (ndarray of k elements);
            unnormalized_heights (ndarray of k elements);
            unnorm_derivatives_left (ndarray of 1 element, or float);
            unnorm_derivatives_right (ndarray of 1 element, or float);
            min_bin_width (float);
            min_bin_height (float).
        '''
        derivatives, slopes, cumwidths, widths, cumheights, heights = coefficient(params)
        a = cumheights[..., :-1]
        b = 3 * cumheights[..., :-1] + derivatives[..., :-1] * widths
        c = 3 * cumheights[..., 1:] - derivatives[..., 1:] * widths
        d = cumheights[..., 1:]

        params = (a, b, c, d)

        return params, cumwidths, widths, cumheights, heights

    @staticmethod
    def initalize(min_bin_width=1e-3, min_bin_height=1e-3, solver='original', quadraticThreshold=1e-7):
        r'''
        initalize parameters
        '''
        return {'splineParams': (min_bin_height, min_bin_width), 'solver': solver, 'quadraticThreshold': quadraticThreshold}


class SteffenSplineFn(GeneralCubicFn):
    '''
    A functional implement of Steffen cubic spline.
    see Steffen, M. A simple method for monotonic interpolation in one dimension. Astronomy and Astrophysics, 239:443, 1990.
    Used in Cubic Spline Flow, arXiv:1906.02145
    '''
    def __init__(self, name="SteffenCubicSpline"):
        super().__init__(name)

    @staticmethod
    def _preprocess(params):
        '''
        params contains:
            unnormalized_widths (ndarray of k elements);
            unnormalized_heights (ndarray of k elements);
            unnorm_derivatives_left (ndarray of 1 element, or float);
            unnorm_derivatives_right (ndarray of 1 element, or float);
            min_bin_width (float);
            min_bin_height (float).
        '''
        derivatives, slopes, cumwidths, widths, cumheights, heights = coefficient(params)
        a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) * widths
        b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) * widths
        c = derivatives[..., :-1] * widths
        d = cumheights[..., :-1]

        params = (a, b, c, d)

        return params, cumwidths, widths, cumheights, heights

    @staticmethod
    def initalize(min_bin_width=1e-3, min_bin_height=1e-3, solver='original', quadraticThreshold=1e-7):
        r'''
        initalize parameters
        '''
        return {'splineParams': (min_bin_height, min_bin_width), 'solver': solver, 'quadraticThreshold': quadraticThreshold}
