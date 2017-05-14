from new_fem import *
from dual_fem import solve_fem as solve_dual_fem
from helpers import new_norm, dual_norm, f_norm, State
from numpy import sqrt
from matplotlib import pyplot as plt
a = 0
b = 1

def draw(state):
    # size, solution, dual_solution, nodes, norm, dual, fn, error

    xs = []
    ys = []
    yds = []
    nodes = state.nodes
    un = state.solution
    dual = state.dual_solution
    size = state.size
    nodes_0 = [0 for i in range(size)]
    nodes_y = [un(nodes[i]) for i in range(size)]
    nodes_yd = [dual(nodes[i]) for i in range(size)]
    size = 1000
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(un(x))
        yds.append(dual(x))
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'b^', xs, yds, 'g', nodes, nodes_yd, 'g^', nodes, nodes_0, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()


# def h_adaptive_fem(m, sigma, f, alpha, _u, nodes, accuracy, states):
#     solution = solve_fem(m=m, sigma=sigma, f=f, alpha=alpha, _u=_u, nodes=nodes)
#     dual_solution = solve_dual_fem(m, sigma, f, alpha, _u, nodes)
#     new_nodes = []
#     straight_norms = []
#     dual_norms = []
#     f_norms = []
#     errors = []
#     size = len(nodes) - 1
#     print(size)
#     fn_full = f_norm(sigma, f, alpha, _u, nodes[0], nodes[-1], nodes[-1])
#     for i in range(size):
#         norm = new_norm(m, sigma, alpha, solution, nodes[i], nodes[i + 1], nodes[-1])
#         dual = dual_norm(m, sigma, alpha, dual_solution, nodes[i], nodes[i + 1], nodes[-1])
#         fn = f_norm(sigma, f, alpha, _u, nodes[i], nodes[i + 1], nodes[-1])
#         straight_norms.append(norm)
#         dual_norms.append(dual)
#         new_nodes.append(nodes[i])
#         f_norms.append(fn)
#         error = abs(fn - (norm + dual))
#         errors.append(error)
#         # if error > accuracy:
#         new_nodes.append((nodes[i] + nodes[i + 1]) / 2)
#     new_nodes.append(nodes[-1])
#     # fn = f_norm(sigma, f, alpha, nodes[0], nodes[-1], nodes[-1])
#     print(fn)
#     print(sum(f_norms))
#     print(alpha * pow(_u, 2))
#     error = sqrt(abs(fn_full - (sum(straight_norms) + sum(dual_norms))) / fn_full)
#     new_state = State(size + 1, solution, dual_solution, nodes, straight_norms, dual_norms, fn_full, error, f_norms, errors)
#     states.append(new_state)
#     # draw(new_state)
#     if len(nodes) < len(new_nodes)and len(nodes) < 150:
#         return h_adaptive_fem(m, sigma, f, alpha, _u, new_nodes, accuracy, states)
#     return states

def h_adaptive_fem(m, sigma, f, alpha, beta, _u, nodes, accuracy, states):
    solution = solve_fem(m=m, sigma=sigma, f=f, alpha=alpha, beta=beta, _u=_u, nodes=nodes)
    dual_solution = solve_dual_fem(m, sigma, f, alpha, beta, _u, nodes)
    error_solution = solve_error(m, sigma, f, alpha, beta, _u, nodes, solution)

    straight_norms = []
    error_norms = []
    size = len(nodes) - 1
    print(size)

    dual_norms = []
    f_norms = []
    norm_full = 0
    error_norm_full = 0
    errors_with_dual = []
    for i in range(size):
        e = new_norm(m, sigma, alpha, beta, error_solution, nodes[i], nodes[i + 1], nodes[-1])
        error_norms.append(e)
        error_norm_full += e

        norm = new_norm(m, sigma, alpha, beta, solution, nodes[i], nodes[i + 1], nodes[-1])
        straight_norms.append(norm)
        norm_full += norm

        dual = dual_norm(m, sigma, alpha, dual_solution, nodes[i], nodes[i + 1], nodes[-1])
        dual_norms.append(dual)
        fn = f_norm(sigma, f, alpha, _u, nodes[i], nodes[i + 1], nodes[-1])
        f_norms.append(fn)

        error = abs(fn - (norm + dual))
        errors_with_dual.append(error)

    new_nodes = []
    errors = []
    for i in range(size):
        new_nodes.append(nodes[i])
        error = sqrt(size * error_norms[i] / (norm_full + error_norm_full))
        errors.append(error)
        # if (error > accuracy):
        new_nodes.append((nodes[i] + nodes[i + 1]) / 2)

    new_nodes.append(nodes[-1])

    fn_full = f_norm(sigma, f, alpha, _u, nodes[0], nodes[-1], nodes[-1])

    error = sqrt(abs(fn_full - (sum(straight_norms) + sum(dual_norms))) / fn_full)
    errors_by_estimator =  sum(errors)
    new_state = State(size + 1, solution, dual_solution, nodes, straight_norms, dual_norms, fn_full, error, f_norms, errors_with_dual, error_norm_full, errors)
    states.append(new_state)
    if len(nodes) < len(new_nodes)and len(nodes) < 70:
        return h_adaptive_fem(m, sigma, f, alpha, beta,  _u, new_nodes, accuracy, states)
    return states