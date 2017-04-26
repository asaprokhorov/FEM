import scipy.integrate as integrate
from scipy.misc import derivative
from numpy import zeros, linalg, sqrt
from helpers import derivative_norm, norm, create_basis, create_bubble_basis, State

def _calculate_matrix_row_elements(i, nodes, phi, m, sigma, alpha):
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
    )[0] + alpha * phi_i(nodes[-1], True) ** 2 / sigma(nodes[-1]) ** 2
    ci_right = None
    if phi_i_right:
        ci_right = integrate.quad(
            lambda x: phi_i(x, True) * phi_i_right(x, True) / sigma(x) + phi_i(
                x) * phi_i_right(x) / m(x),
            xi, xi_right
        )[0]

    return ci_left, ci, ci_right


def _calculate_vector_element(i, nodes, phi, sigma, f, alpha):
    xi_left = nodes[i - 1] if i > 0 else nodes[i]
    xi_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[i]
    return integrate.quad(lambda x: f(x) * phi[i](x, True) / sigma(x), xi_left, xi_right)[0] + alpha * phi[i](
        nodes[-1], True) * f(nodes[-1]) / sigma(nodes[-1]) ** 2


def _create_matrix(nodes, phi, m, sigma, f, alpha, _u):
    size = len(nodes)
    matrix = zeros((size, size))
    b = zeros(size)
    for i in range(size):
        ci_left, ci, ci_right = _calculate_matrix_row_elements(i, nodes, phi, m, sigma, alpha)
        if ci_left:
            matrix[i][i - 1] = ci_left
        matrix[i][i] = ci
        if ci_right:
            matrix[i][i + 1] = ci_right
        b[i] = _calculate_vector_element(i, nodes, phi, sigma, f, alpha)
    return matrix, b


def solve_fem(m, sigma, f, alpha, _u, nodes):
    phi = create_basis(nodes)
    matrix, b = _create_matrix(nodes=nodes, phi=phi, m=m, sigma=sigma, f=f, alpha=alpha,
                               _u=_u)
    solution = linalg.solve(matrix, b)
    # print(solution)
    return lambda x: sum([phi[i](x) * solution[i] for i in range(len(solution))])


def _calculate_error_matrix_row_element(i, nodes, bubble_basis, m, sigma, alpha):
    x_i = nodes[i]
    x_i_next = nodes[i + 1]
    basis = bubble_basis[i]
    return integrate.quad(lambda x: basis(x, True) ** 2 / sigma(x) + basis(x) ** 2 / m(x), x_i, x_i_next)[
               0] + alpha * basis(
        nodes[-1], True) ** 2 / sigma(nodes[-1]) ** 2


def _calculate_error_matrix_vector_element(i, nodes, bubble_basis, m, sigma, f, alpha, _u, solution):
    x_i = nodes[i]
    x_i_next = nodes[i + 1]
    basis = bubble_basis[i]
    return integrate.quad(
        lambda x: basis(x, True) * (f(x) - derivative(solution, x, dx=1e-6)) / sigma(x) - basis(x) * solution(x) / m(x),
        x_i, x_i_next)[0] + alpha * (
    basis(nodes[-1]) * f(nodes[-1]) - basis(nodes[-1], True) * derivative(solution, nodes[-1], dx=1e-6)) / sigma(
        nodes[-1]) ** 2


def _calculate_error_norms(nodes, m, sigma, alpha, f, _u, solution):
    bubble_basis = create_bubble_basis(nodes)
    size = len(nodes) - 1
    e_h = []
    e_l = []
    solution_h = []
    solution_l = []
    for i in range(size):
        e_i = _calculate_error_matrix_row_element(i, nodes, bubble_basis, m, sigma, alpha)

        f_i = _calculate_error_matrix_vector_element(i, nodes, bubble_basis, m, sigma, f, alpha, _u, solution)
        coefficient = f_i / e_i
        e_h.append(coefficient ** 2 * derivative_norm(bubble_basis[i], nodes[i], nodes[i + 1]) ** 2)
        e_l.append(coefficient ** 2 * norm(bubble_basis[i], nodes[i], nodes[i + 1]) ** 2)
        solution_h.append(derivative_norm(solution, nodes[i], nodes[i + 1]) ** 2)
        solution_l.append(norm(solution, nodes[i], nodes[i + 1]) ** 2)
    return e_h, e_l, sum(solution_h), sum(solution_l)


def h_adaptive_fem(m, sigma, f, alpha, _u, nodes, accuracy, states):
    solution = solve_fem(m=m, sigma=sigma, f=f, alpha=alpha, _u=_u, nodes=nodes)
    e_h, e_l, uh_h, uh_l = _calculate_error_norms(nodes=nodes, m=m, sigma=sigma, alpha=alpha, f=f, _u=_u,
                                                  solution=solution)
    size = len(nodes) - 1
    new_state = State(size + 1, sqrt(uh_l), sqrt(sum(e_l)), sqrt(uh_h), sqrt(sum(e_h)), solution, nodes)
    states.append(new_state)
    new_nodes = []
    sum_e_h = sum(e_h)
    print("size: {0}".format(size))
    for i in range(size):
        new_nodes.append(nodes[i])
        deviation = sqrt(size * e_h[i] / (sum_e_h + uh_h))
        if deviation > accuracy:
            new_nodes.append((nodes[i] + nodes[i + 1]) / 2)
    new_nodes.append(nodes[-1])
    if len(nodes) < len(new_nodes):
        return h_adaptive_fem(m, sigma, f, alpha, _u, new_nodes, accuracy, states)
    return states