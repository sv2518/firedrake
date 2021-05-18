import numpy
from functools import partial, singledispatch

import FIAT
import ufl
from ufl.algorithms import extract_arguments

from pyop2 import op2

from tsfc.finatinterface import create_base_element, as_fiat_cell
from tsfc import compile_expression_dual_evaluation

import gem
import finat

import firedrake
from firedrake import utils
from firedrake.adjoint import annotate_interpolate

__all__ = ("interpolate", "Interpolator")


def interpolate(expr, V, subset=None, access=op2.WRITE):
    """Interpolate an expression onto a new function in V.

    :arg expr: an :class:`.Expression`.
    :arg V: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Function`).
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
        interpolation over.
    :kwarg access: The access descriptor for combining updates to shared dofs.
    :returns: a new :class:`.Function` in the space ``V`` (or ``V`` if
        it was a Function).

    .. note::

       If you use an access descriptor other than ``WRITE``, the
       behaviour of interpolation is changes if interpolating into a
       function space, or an existing function. If the former, then
       the newly allocated function will be initialised with
       appropriate values (e.g. for MIN access, it will be initialised
       with MAX_FLOAT). On the other hand, if you provide a function,
       then it is assumed that its values should take part in the
       reduction (hence using MIN will compute the MIN between the
       existing values and any new values).

    .. note::

       If you find interpolating the same expression again and again
       (for example in a time loop) you may find you get better
       performance by using an :class:`Interpolator` instead.

    """
    return Interpolator(expr, V, subset=subset, access=access).interpolate()


class Interpolator(object):
    """A reusable interpolation object.

    :arg expr: The expression to interpolate.
    :arg V: The :class:`.FunctionSpace` or :class:`.Function` to
        interpolate into.
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
        interpolation over.
    :kwarg freeze_expr: Set to True to prevent the expression being
        re-evaluated on each call.

    This object can be used to carry out the same interpolation
    multiple times (for example in a timestepping loop).

    .. note::

       The :class:`Interpolator` holds a reference to the provided
       arguments (such that they won't be collected until the
       :class:`Interpolator` is also collected).

    """
    def __init__(self, expr, V, subset=None, freeze_expr=False, access=op2.WRITE):
        try:
            self.callable, arguments = make_interpolator(expr, V, subset, access)
        except FIAT.hdiv_trace.TraceError:
            raise NotImplementedError("Can't interpolate onto traces sorry")
        self.arguments = arguments
        self.nargs = len(arguments)
        self.freeze_expr = freeze_expr
        self.expr = expr
        self.V = V

    @annotate_interpolate
    def interpolate(self, *function, output=None, transpose=False):
        """Compute the interpolation.

        :arg function: If the expression being interpolated contains an
            :class:`ufl.Argument`, then the :class:`.Function` value to
            interpolate.
        :kwarg output: Optional. A :class:`.Function` to contain the output.
        :kwarg transpose: Set to true to apply the transpose (adjoint) of the
              interpolation operator.
        :returns: The resulting interpolated :class:`.Function`.
        """
        if transpose and not self.nargs:
            raise ValueError("Can currently only apply transpose interpolation with arguments.")
        if self.nargs != len(function):
            raise ValueError("Passed %d Functions to interpolate, expected %d"
                             % (len(function), self.nargs))
        try:
            assembled_interpolator = self.frozen_assembled_interpolator
            copy_required = True
        except AttributeError:
            assembled_interpolator = self.callable()
            copy_required = False  # Return the original
            if self.freeze_expr:
                if self.nargs:
                    # Interpolation operator
                    self.frozen_assembled_interpolator = assembled_interpolator
                else:
                    # Interpolation action
                    self.frozen_assembled_interpolator = assembled_interpolator.copy()

        if self.nargs:
            function, = function
            if transpose:
                mul = assembled_interpolator.handle.multTranspose
                V = self.arguments[0].function_space()
            else:
                mul = assembled_interpolator.handle.mult
                V = self.V
            result = output or firedrake.Function(V)
            with function.dat.vec_ro as x, result.dat.vec_wo as out:
                mul(x, out)
            return result

        else:
            if output:
                output.assign(assembled_interpolator)
                return output
            if isinstance(self.V, firedrake.Function):
                if copy_required:
                    self.V.assign(assembled_interpolator)
                return self.V
            else:
                if copy_required:
                    return assembled_interpolator.copy()
                else:
                    return assembled_interpolator


def make_interpolator(expr, V, subset, access):
    assert isinstance(expr, ufl.classes.Expr)

    if isinstance(expr, firedrake.Expression):
        arguments = ()
    else:
        arguments = extract_arguments(expr)
    if len(arguments) == 0:
        if isinstance(V, firedrake.Function):
            f = V
            V = f.function_space()
        else:
            f = firedrake.Function(V)
            if access in {firedrake.MIN, firedrake.MAX}:
                finfo = numpy.finfo(f.dat.dtype)
                if access == firedrake.MIN:
                    val = firedrake.Constant(finfo.max)
                else:
                    val = firedrake.Constant(finfo.min)
                f.assign(val)
        tensor = f.dat
    elif len(arguments) == 1:
        if isinstance(V, firedrake.Function):
            raise ValueError("Cannot interpolate an expression with an argument into a Function")
        argfs = arguments[0].function_space()
        argfs_map = argfs.cell_node_map()
        if argfs_map is None or argfs_map.iterset != V.cell_node_map().iterset:
            if isinstance(V.ufl_domain().topology, firedrake.mesh.VertexOnlyMeshTopology):
                # Compose a vertex-cell to parent-cell-function-space-node map
                argfs_map = compose_map_and_cache(V.ufl_domain().cell_parent_cell_map, argfs_map)
        sparsity = op2.Sparsity((V.dof_dset, argfs.dof_dset),
                                ((V.cell_node_map(), argfs_map),),
                                name="%s_%s_sparsity" % (V.name, argfs.name),
                                nest=False,
                                block_sparse=True)
        tensor = op2.Mat(sparsity)
        f = tensor
    else:
        raise ValueError("Cannot interpolate an expression with %d arguments" % len(arguments))

    # Make sure we have an expression of the right length i.e. a value for
    # each component in the value shape of each function space
    dims = [numpy.prod(fs.ufl_element().value_shape(), dtype=int)
            for fs in V]
    loops = []
    if numpy.prod(expr.ufl_shape, dtype=int) != sum(dims):
        raise RuntimeError('Expression of length %d required, got length %d'
                           % (sum(dims), numpy.prod(expr.ufl_shape, dtype=int)))

    if not isinstance(expr, firedrake.Expression):
        if len(V) > 1:
            raise NotImplementedError(
                "UFL expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, tensor, expr, subset, arguments, access))
    elif hasattr(expr, 'eval'):
        if len(V) > 1:
            raise NotImplementedError(
                "Python expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, tensor, expr, subset, arguments, access))
    else:
        raise ValueError("Don't know how to interpolate a %r" % expr)

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f), arguments


@utils.known_pyop2_safe
def _interpolator(V, tensor, expr, subset, arguments, access):
    try:
        to_element = create_base_element(V.ufl_element())
    except KeyError:
        # FInAT only elements
        raise NotImplementedError("Don't know how to create FIAT element for %s" % V.ufl_element())

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")

    if len(expr.ufl_shape) != len(V.ufl_element().value_shape()):
        raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                           % (len(expr.ufl_shape), len(V.ufl_element().value_shape())))

    if expr.ufl_shape != V.ufl_element().value_shape():
        raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                           % (expr.ufl_shape, V.ufl_element().value_shape()))

    target_mesh = V.ufl_domain()
    try:
        trans_mesh = (
            expr.ufl_domain() is not None  # Constant expr
            and expr.ufl_domain() != V.mesh()  # Coming from a different domain
            and isinstance(target_mesh.topology,  # Target domain is the one we support
                           firedrake.mesh.VertexOnlyMeshTopology)
        )
    except AttributeError:
        # Have python Expression
        trans_mesh = False

    if trans_mesh:
        if target_mesh.geometric_dimension() != expr.ufl_domain().geometric_dimension():
            raise ValueError("Cannot interpolate onto a VertexOnlyMesh of a different geometric dimension")
        # For trans-mesh interpolation we use a FInAT QuadratureElement as the
        # (base) target element with runtime point set expressions as their
        # quadrature rule point set and appropriate weights
        to_element = rebuild(to_element, expr)
        # The source domain is on the expression not the FunctionSpace/Function
        # we are interpolating into/onto
        source_coords_coeff = expr.ufl_domain().coordinates
        source_domain = expr.ufl_domain()
    else:
        # The source domain is that of the FunctionSpace/Function we are
        # interpolating into/onto
        source_coords_coeff = target_mesh.coordinates
        source_domain = V.mesh()

    parameters = {}
    parameters['scalar_type'] = utils.ScalarType

    if not isinstance(expr, firedrake.Expression):
        if expr.ufl_domain() and expr.ufl_domain() != V.mesh() and not trans_mesh:
            raise NotImplementedError("Interpolation onto the target mesh not supported.")
        ast, oriented, needs_cell_sizes, coefficients, first_coeff_fake_coords, _ = compile_expression_dual_evaluation(expr, to_element,
                                                                                                                       domain=source_domain,
                                                                                                                       parameters=parameters,
                                                                                                                       coffee=False)
        kernel = op2.Kernel(ast, ast.name, requires_zeroed_output_arguments=True)
    elif hasattr(expr, "eval"):
        to_pts = []
        for dual in to_element.fiat_equivalent.dual_basis():
            if not isinstance(dual, FIAT.functional.PointEvaluation):
                raise NotImplementedError("Can only interpolate Python kernels with Lagrange elements")
            pts, = dual.pt_dict.keys()
            to_pts.append(pts)
        kernel, oriented, needs_cell_sizes, coefficients = compile_python_kernel(expr, to_pts, to_element, V, source_coords_coeff)
        first_coeff_fake_coords = False
    else:
        raise RuntimeError("Attempting to evaluate an Expression which has no value.")

    cell_set = target_mesh.coordinates.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset
    parloop_args = [kernel, cell_set]

    if first_coeff_fake_coords:
        # Replace with real source_coords_coeff
        coefficients[0] = source_coords_coeff

    if trans_mesh:
        # Add the coordinates of the target mesh quadrature points in the
        # source mesh's reference cell as an extra argument for the inner loop.
        # (with a vertex only mesh this is a single point for each vertex cell)
        coefficients = coefficients + [target_mesh.reference_coordinates]
        try:
            cell_node_map = arguments[0].function_space().cell_node_map()
        except IndexError:
            # should have had coefficients before adding reference coords coeff
            assert len(coefficients) > 1
            cell_node_map = coefficients[0].cell_node_map()
        cell_parent_cell_fs_node_map = compose_map_and_cache(target_mesh.cell_parent_cell_map,
                                                             cell_node_map)

    if tensor in set((c.dat for c in coefficients)):
        output = tensor
        tensor = op2.Dat(tensor.dataset)
        if access is not op2.WRITE:
            copyin = (partial(output.copy, tensor), )
        else:
            copyin = ()
        copyout = (partial(tensor.copy, output), )
    else:
        copyin = ()
        copyout = ()
    if isinstance(tensor, op2.Global):
        parloop_args.append(tensor(access))
    elif isinstance(tensor, op2.Dat):
        parloop_args.append(tensor(access, V.cell_node_map()))
    else:
        assert access == op2.WRITE  # Other access descriptors not done for Matrices.
        if trans_mesh:
            cmap = cell_parent_cell_fs_node_map
        else:
            cmap = arguments[0].function_space().cell_node_map()
        parloop_args.append(tensor(op2.WRITE, (V.cell_node_map(), cmap)))
    if oriented:
        co = target_mesh.cell_orientations()
        parloop_args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = target_mesh.cell_sizes
        parloop_args.append(cs.dat(op2.READ, cs.cell_node_map()))
    for coefficient in coefficients:
        get_composed_map = (
            trans_mesh
            and (
                isinstance(target_mesh.topology,
                           firedrake.mesh.VertexOnlyMeshTopology)
                and not isinstance(
                    coefficient.function_space().mesh().topology,
                    firedrake.mesh.VertexOnlyMeshTopology)
            )
        )
        if get_composed_map:
            m_ = compose_map_and_cache(target_mesh.cell_parent_cell_map, coefficient.cell_node_map())
            parloop_args.append(coefficient.dat(op2.READ, m_))
        else:
            m_ = coefficient.cell_node_map()
            parloop_args.append(coefficient.dat(op2.READ, m_))

    for o in coefficients:
        domain = o.ufl_domain()
        if domain is not None and domain.topology != target_mesh.topology and not trans_mesh:
            raise NotImplementedError("Interpolation onto the target mesh not supported.")

    parloop = op2.ParLoop(*parloop_args).compute
    if isinstance(tensor, op2.Mat):
        return parloop, tensor.assemble
    else:
        return copyin + (parloop, ) + copyout


@singledispatch
def rebuild(element, expr):
    raise ValueError("Not expecting %s" % element)


@rebuild.register(finat.DiscontinuousLagrange)
def rebuild_dg(element, expr):
    # Create a quadrature element for a runtime point with weight 1
    # to represent the runtime tabulated point, on the expression's
    # mesh reference cell, onto which we interpolate
    if element.degree != 0:
        raise ValueError("Can only runtime interpolate onto DG0")
    expr_tdim = expr.ufl_domain().topological_dimension()
    # gem.Variable name starting with rt_ forces TSFC runtime tabulation
    runtime_point = gem.Variable('rt_X', (expr_tdim,))
    rule_pointset = finat.point_set.UnknownPointSingleton(runtime_point)
    try:
        expr_fiat_cell = as_fiat_cell(expr.ufl_element().cell())
    except AttributeError:
        # expression must be pure function of spatial coordinates so
        # domain has correct ufl cell
        expr_fiat_cell = as_fiat_cell(expr.ufl_domain().ufl_cell())
    rule = finat.quadrature.QuadratureRule(rule_pointset, weights=[1.])
    return finat.QuadratureElement(expr_fiat_cell, None, rule=rule)


@rebuild.register(finat.TensorFiniteElement)
def rebuild_te(element, expr):
    return finat.TensorFiniteElement(rebuild(element.base_element. expr),
                                     element.shape,
                                     transpose=element._transpose)


def composed_map(map1, map2):
    """
    Manually build a :class:`PyOP2.Map` from the iterset of map1 to the
    toset of map2.

    :arg map1: The map with the desired iterset
    :arg map2: The map with the desired toset

    :returns:  The composed map

    Requires that `map1.toset == map2.iterset`.
    Only currently implemented for `map1.arity == 1`
    """
    if map1.toset != map2.iterset:
        raise ValueError("Cannot compose a map where the intermediate sets do not match!")
    if map1.arity != 1:
        raise NotImplementedError("Can only currently build composed maps where map1.arity == 1")
    iterset = map1.iterset
    toset = map2.toset
    arity = map2.arity
    values = map2.values[map1.values].reshape(iterset.size, arity)
    assert values.shape == (iterset.size, arity)
    return op2.Map(iterset, toset, arity, values)


def compose_map_and_cache(map1, map2):
    """
    Retrieve a composed :class:`PyOP2.Map` map from the cache of map1
    using map2 as the cache key. The composed map maps from the iterset
    of map1 to the toset of map2. Calls `composed_map` and caches the
    result on map1 if the composed map is not found.

    :arg map1: The map with the desired iterset from which the result is
        retrieved or cached
    :arg map2: The map with the desired toset

    :returns:  The composed map

    See also `composed_map`.
    """
    cache_key = hash((map2, "composed"))
    try:
        cmap = map1._cache[cache_key]
    except KeyError:
        cmap = composed_map(map1, map2)
        map1._cache[cache_key] = cmap
    return cmap


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *arguments: None
        self.ufl_domain = lambda: None


def compile_python_kernel(expression, to_pts, to_element, fs, coords):
    """Produce a :class:`PyOP2.Kernel` wrapping the eval method on the
    function provided."""

    coords_space = coords.function_space()
    coords_element = create_base_element(coords_space.ufl_element()).fiat_equivalent

    X_remap = list(coords_element.tabulate(0, to_pts).values())[0]

    # The par_loop will just pass us arguments, since it doesn't
    # know about keyword arguments at all so unpack into a dict that we
    # can pass to the user's eval method.
    def kernel(output, x, *arguments):
        kwargs = {}
        for (slot, _), arg in zip(expression._user_args, arguments):
            kwargs[slot] = arg
        X = numpy.dot(X_remap.T, x)

        for i in range(len(output)):
            # Pass a slice for the scalar case but just the
            # current vector in the VFS case. This ensures the
            # eval method has a Dolfin compatible API.
            expression.eval(output[i:i+1, ...] if numpy.ndim(output) == 1 else output[i, ...],
                            X[i:i+1, ...] if numpy.ndim(X) == 1 else X[i, ...], **kwargs)

    coefficients = [coords]
    for _, arg in expression._user_args:
        coefficients.append(GlobalWrapper(arg))
    return kernel, False, False, tuple(coefficients)
