import scipy.integrate as integrate
from scipy.misc import derivative
from numpy import zeros, linalg, sqrt
from helpers import create_basis, create_bubble_basis, State


def _calculate_matrix_row_elements(i, nodes, basis, m, sigma, alpha, beta):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi = nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    phi_i_left = basis[i - 1] if i > 0 else None
    phi_i = basis[i]
    phi_i_right = basis[i + 1] if i < len(basis) - 1 else None
    ci_left = None
    if phi_i_left:
        # ci_left = integrate.quad(
        #     lambda x: m(x) * phi_i_left(x, True) * phi_i(x, True) + sigma(x) * phi_i_left(
        #         x) * phi_i(x),
        #     xi_left, xi)[0]

        ci_left = integrate.quad(
            lambda x: m(x) * phi_i_left(x, True) * phi_i(x, True) + sigma(x) * phi_i_left(
                x) * phi_i(x) + beta(x) * phi_i_left(x, True) * phi_i(x),
            xi_left, xi)[0]

    # ci = integrate.quad(
    #     lambda x: m(x) * phi_i(x, True) ** 2 + sigma(x) * phi_i(x) ** 2, xi_left, xi_right
    # )[0] + alpha * phi_i(nodes[-1]) ** 2

    ci = integrate.quad(
        lambda x: m(x) * phi_i(x, True) ** 2 + sigma(x) * phi_i(x) ** 2 + beta(x) * phi_i(x, True) * phi_i(x), xi_left,
        xi_right
    )[0] + alpha * phi_i(nodes[-1]) ** 2

    ci_right = None
    if phi_i_right:
        ci_right = integrate.quad(
            lambda x: m(x) * phi_i(x, True) * phi_i_right(x, True) + sigma(x) * phi_i(
                x) * phi_i_right(x) + beta(x) * phi_i(x, True) * phi_i_right(x),
            xi, xi_right
        )[0]

    return ci_left, ci, ci_right


def _calculate_vector_element(i, nodes, basis, f, alpha, beta, _u):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    return integrate.quad(lambda x: f(x) * basis[i](x), xi_left, xi_right)[0] + alpha * _u * basis[i](nodes[-1])


def _create_matrix(nodes, basis, m, sigma, f, alpha, beta, _u):
    size = len(nodes)
    matrix = zeros((size, size))
    b = zeros(size)
    for i in range(size):
        ci_left, ci, ci_right = _calculate_matrix_row_elements(i, nodes, basis, m, sigma, alpha, beta)
        if ci_left:
            matrix[i][i - 1] = ci_left
        matrix[i][i] = ci
        if ci_right:
            matrix[i][i + 1] = ci_right
        b[i] = _calculate_vector_element(i, nodes, basis, f, alpha, beta, _u)
        matrix[0][0] = 10 ** 15
    return matrix, b


def solve_fem(m, sigma, f, alpha, beta, _u, nodes):
    basis = create_basis(nodes)
    matrix, b = _create_matrix(nodes=nodes, basis=basis, m=m, sigma=sigma, f=f, alpha=alpha, beta=beta,
                               _u=_u)
    solution = linalg.solve(matrix, b)
    return lambda x: sum([basis[i](x) * solution[i] for i in range(len(solution))])


def solve_error(m, sigma, f, alpha, beta, _u, nodes, solution):
    basis = create_bubble_basis(nodes)
    size = len(nodes) - 1
    c = []
    for i in range(size):
        matrix_element = \
            integrate.quad(lambda x: m(x) * basis[i](x, True) ** 2 + sigma(x) * basis[i](x) ** 2, nodes[i],
                           nodes[i + 1])[
                0] + \
            alpha * basis[i](nodes[-1]) ** 2

        vector_element = integrate.quad(
            lambda x: f(x) * basis[i](x) - m(x) * derivative(solution, x, dx=1e-6) * basis[i](x, True) + beta(
                x) * derivative(solution, x, dx=1e-6) * basis[i](x) - sigma(
                x) * solution(x) * basis[i](x), nodes[i], nodes[i + 1])[0] + alpha * basis[i](nodes[-1]) * (
        _u - solution(nodes[-1]))
        c.append(vector_element / matrix_element)
    return lambda x: sum([basis[i](x) * c[i] for i in range(len(c))])
