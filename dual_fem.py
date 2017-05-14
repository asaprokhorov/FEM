import scipy.integrate as integrate
from scipy.misc import derivative
from numpy import zeros, linalg, sqrt, pi
from helpers import create_basis, create_bubble_basis, State

def _calculate_matrix_row_elements(i, nodes, phi, m, sigma, alpha, beta):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi = nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    phi_i_left = phi[i - 1] if i > 0 else None
    phi_i = phi[i]
    phi_i_right = phi[i + 1] if i < len(phi) - 1 else None
    ci_left = None
    if phi_i_left:
        ci_left = integrate.quad(
            lambda x: phi_i_left(x, True) * phi_i(x, True) / sigma(x) + phi_i_left(
                x) * phi_i(x) / m(x),
            xi_left, xi)[0]
    ci = integrate.quad(
        lambda x: phi_i(x, True) ** 2 / sigma(x) + phi_i(x) ** 2 / m(x), xi_left, xi_right
    )[0] + phi_i(nodes[-1]) ** 2 / alpha
    ci_right = None
    if phi_i_right:
        ci_right = integrate.quad(
            lambda x: phi_i(x, True) * phi_i_right(x, True) / sigma(x) + phi_i(
                x) * phi_i_right(x) / m(x),
            xi, xi_right
        )[0]

    return ci_left, ci, ci_right


def _calculate_vector_element(i, nodes, phi, sigma, f, alpha, _u):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    return integrate.quad(lambda x: f(x) * phi[i](x, True) / sigma(x), xi_left, xi_right)[0] - phi[i](
        nodes[-1]) * _u


def _create_matrix(nodes, phi, m, sigma, f, alpha, beta, _u):
    size = len(nodes)
    matrix = zeros((size, size))
    b = zeros(size)
    for i in range(size):
        ci_left, ci, ci_right = _calculate_matrix_row_elements(i, nodes, phi, m, sigma, alpha, beta)
        if ci_left:
            matrix[i][i - 1] = ci_left
        matrix[i][i] = ci
        if ci_right:
            matrix[i][i + 1] = ci_right
        b[i] = _calculate_vector_element(i, nodes, phi, sigma, f, alpha, _u)
    return matrix, b


def solve_fem(m, sigma, f, alpha, beta, _u, nodes):
    phi = create_basis(nodes)
    matrix, b = _create_matrix(nodes=nodes, phi=phi, m=m, sigma=sigma, f=f, alpha=alpha, beta=beta,
                               _u=_u)
    solution = linalg.solve(matrix, b)
    return lambda x: sum([phi[i](x) * solution[i] for i in range(len(solution))])