import scipy.integrate as integrate
from scipy.misc import derivative
from numpy import zeros, linalg, sqrt, pi
from helpers import create_basis, create_bubble_basis, State


def _calculate_matrix_row_elements(i, nodes, phi, m, beta, sigma, alpha):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi = nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    phi_i_left = phi[i - 1] if i > 0 else None
    phi_i = phi[i]
    phi_i_right = phi[i + 1] if i < len(phi) - 1 else None
    ci_left = None
    def bilinear_form(f_i, f_i_next, begin, end):

        return integrate.quad(
            lambda x: f_i(x, True) * f_i_next(x, True) / sigma(x) + f_i(
                x) * f_i_next(x) / m(x) -
                                  beta(x) * f_i(x, True) * f_i_next(x) / ( m(x) * sigma(x)),
            begin, end)[0] + f_i(nodes[-1]) * f_i_next(nodes[-1]) / alpha

    if phi_i_left:
        ci_left = bilinear_form(phi_i, phi_i_left, xi_left, xi)

    ci = bilinear_form(phi_i, phi_i, xi_left, xi_right)
    ci_right = None

    if phi_i_right:
        ci_right = bilinear_form(phi_i, phi_i_right, xi, xi_right)

    return ci_left, ci, ci_right


def _calculate_vector_element(i, nodes, phi, m, beta, sigma, f, alpha, _u):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    # return integrate.quad(lambda x: f(x) * phi[i](x, True) / sigma(x), xi_left, xi_right)[0] - phi[i](
    #     nodes[-1]) * _u
    return \
        integrate.quad(lambda x: f(x) * phi[i](x, True) / sigma(x), #- beta(x) * f(x) * phi[i](x) / (2 * m(x) * sigma(x)),
                       xi_left, xi_right)[0] - phi[i](nodes[-1]) * _u


def _create_matrix(nodes, phi, m, beta, sigma, f, alpha, _u):
    size = len(nodes)
    matrix = zeros((size, size))
    b = zeros(size)
    for i in range(size):
        ci_left, ci, ci_right = _calculate_matrix_row_elements(i, nodes, phi, m, beta, sigma, alpha)
        if ci_left:
            matrix[i][i - 1] = ci_left
        matrix[i][i] = ci
        if ci_right:
            matrix[i][i + 1] = ci_right
        b[i] = _calculate_vector_element(i, nodes, phi, m, beta, sigma, f, alpha, _u)
    return matrix, b


def solve_fem(m, beta, sigma, f, alpha, _u, nodes):
    phi = create_basis(nodes)
    matrix, b = _create_matrix(nodes=nodes, phi=phi, m=m, beta=beta, sigma=sigma, f=f, alpha=alpha,
                               _u=_u)
    solution = linalg.solve(matrix, b)
    return lambda x: sum([phi[i](x) * solution[i] for i in range(len(solution))])
