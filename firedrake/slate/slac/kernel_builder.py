import numpy as np
from coffee import base as ast

from collections import OrderedDict, Counter, namedtuple

from firedrake.slate.slac.utils import traverse_dags, Transformer
from firedrake.utils import cached_property

from tsfc.finatinterface import create_element
from ufl import MixedElement, Coefficient, FunctionSpace
import loopy

from loopy.symbolic import SubArrayRef
import pymbolic.primitives as pym

from functools import singledispatch
import firedrake.slate.slate as slate
from firedrake.slate.slac.tsfc_driver import compile_terminal_form
from firedrake.slate.slac.kernel_settings import knl_counter

from tsfc import kernel_args
from tsfc.finatinterface import create_element
from tsfc.loopy import create_domains, assign_dtypes
from pyop2 import configuration

from pytools import UniqueNameGenerator

CoefficientInfo = namedtuple("CoefficientInfo",
                             ["space_index",
                              "offset_index",
                              "shape",
                              "vector",
                              "local_temp"])
CoefficientInfo.__doc__ = """\
Context information for creating coefficient temporaries.

:param space_index: An integer denoting the function space index.
:param offset_index: An integer denoting the starting position in
                     the vector temporary for assignment.
:param shape: A singleton with an integer describing the shape of
              the coefficient temporary.
:param vector: The :class:`slate.AssembledVector` containing the
               relevant data to be placed into the temporary.
:param local_temp: The local temporary for the coefficient vector.
"""


class LayerCountKernelArg(kernel_args.KernelArg):
    ...


class CellFacetKernelArg(kernel_args.KernelArg):
    ...


class LocalKernelBuilder(object):
    """The primary helper class for constructing cell-local linear
    algebra kernels from Slate expressions.

    This class provides access to all temporaries and subkernels associated
    with a Slate expression. If the Slate expression contains nodes that
    require operations on already assembled data (such as the action of a
    slate tensor on a `ufl.Coefficient`), this class provides access to the
    expression which needs special handling.

    Instructions for assembling the full kernel AST of a Slate expression is
    provided by the method `construct_ast`.
    """

    # Relevant symbols/information needed for kernel construction
    # defined below
    coord_sym = ast.Symbol("coords")
    cell_orientations_sym = ast.Symbol("cell_orientations")
    cell_facet_sym = ast.Symbol("cell_facets")
    it_sym = ast.Symbol("i0")
    mesh_layer_sym = ast.Symbol("layer")
    mesh_layer_count_sym = ast.Symbol("layer_count")
    cell_size_sym = ast.Symbol("cell_sizes")

    # Supported integral types
    supported_integral_types = [
        "cell",
        "interior_facet",
        "exterior_facet",
        # The "interior_facet_horiz" measure is separated into two parts:
        # "top" and "bottom"
        "interior_facet_horiz_top",
        "interior_facet_horiz_bottom",
        "interior_facet_vert",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "exterior_facet_vert"
    ]

    # Supported subdomain types
    supported_subdomain_types = ["subdomains_exterior_facet",
                                 "subdomains_interior_facet"]

    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the LocalKernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
            TSFC when constructing subkernels associated with the expression.
        """
        assert isinstance(expression, slate.TensorBase)

        # Collect terminals, expressions, and reference counts
        temps = OrderedDict()
        coeff_vecs = OrderedDict()
        seen_coeff = set()
        expression_dag = list(traverse_dags([expression]))
        counter = Counter([expression])
        for tensor in expression_dag:
            counter.update(tensor.operands)

            # Terminal tensors will always require a temporary.
            if isinstance(tensor, slate.Tensor):
                temps.setdefault(tensor, ast.Symbol("T%d" % len(temps)))

            # 'AssembledVector's will always require a coefficient temporary.
            if isinstance(tensor, slate.AssembledVector):
                function = tensor._function

                def dimension(e):
                    return create_element(e).space_dimension()

                # Ensure coefficient temporaries aren't duplicated
                if function not in seen_coeff:
                    if type(function.ufl_element()) == MixedElement:
                        shapes = [dimension(element) for element in function.ufl_element().sub_elements()]
                    else:
                        shapes = [dimension(function.ufl_element())]

                    # Local temporary
                    local_temp = ast.Symbol("VecTemp%d" % len(seen_coeff))

                    offset = 0
                    for i, shape in enumerate(shapes):
                        cinfo = CoefficientInfo(space_index=i,
                                                offset_index=offset,
                                                shape=(sum(shapes), ),
                                                vector=tensor,
                                                local_temp=local_temp)
                        coeff_vecs.setdefault(shape, []).append(cinfo)
                        offset += shape

                    seen_coeff.add(function)

        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.temps = temps
        self.ref_counter = counter
        self.expression_dag = expression_dag
        self.coefficient_vecs = coeff_vecs
        self._setup()

    @cached_property
    def terminal_flops(self):
        flops = 0
        nfacets = self.expression.ufl_domain().ufl_cell().num_facets()
        for ctx in self.context_kernels:
            itype = ctx.original_integral_type
            for k in ctx.tsfc_kernels:
                kinfo = k.kinfo
                if itype == "cell":
                    flops += kinfo.kernel.num_flops
                elif itype.startswith("interior_facet"):
                    # Executed once per facet (approximation)
                    flops += kinfo.kernel.num_flops * nfacets
                else:
                    # Exterior facets basically contribute zero flops
                    pass
        return int(flops)

    @cached_property
    def expression_flops(self):
        @singledispatch
        def _flops(expr):
            raise AssertionError("Unhandled type %r" % type(expr))

        @_flops.register(slate.AssembledVector)
        @_flops.register(slate.Block)
        @_flops.register(slate.Tensor)
        @_flops.register(slate.Transpose)
        @_flops.register(slate.Negative)
        def _flops_none(expr):
            return 0

        @_flops.register(slate.Factorization)
        def _flops_factorization(expr):
            m, n = expr.shape
            decomposition = expr.decomposition
            # Extracted from Golub & Van Loan
            # These all ignore lower-order terms...
            if decomposition in {"PartialPivLU", "FullPivLU"}:
                return 2/3 * n**3
            elif decomposition in {"LLT", "LDLT"}:
                return (1/3)*n**3
            elif decomposition in {"HouseholderQR", "ColPivHouseholderQR", "FullPivHouseholderQR"}:
                return 4/3 * n**3
            elif decomposition in {"BDCSVD", "JacobiSVD"}:
                return 12 * n**3
            else:
                # Don't know, but don't barf just because of it.
                return 0

        @_flops.register(slate.Inverse)
        def _flops_inverse(expr):
            m, n = expr.shape
            assert m == n
            # Assume LU factorisation
            return (2/3)*n**3

        @_flops.register(slate.Add)
        def _flops_add(expr):
            return int(np.prod(expr.shape))

        @_flops.register(slate.Mul)
        def _flops_mul(expr):
            A, B = expr.operands
            *rest_a, col = A.shape
            _, *rest_b = B.shape
            return 2*col*int(np.prod(rest_a))*int(np.prod(rest_b))

        @_flops.register(slate.Solve)
        def _flops_solve(expr):
            Afac, B = expr.operands
            _, *rest = B.shape
            m, n = Afac.shape
            # Forward elimination + back sub on factorised matrix
            return (m*n + n**2)*int(np.prod(rest))

        return int(sum(map(_flops, traverse_dags([self.expression]))))

    def _setup(self):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC and creates templated function calls
        conforming to the Eigen-C++ template library standard.
        This function also collects any information regarding orientations
        and extra include directories.
        """
        transformer = Transformer()
        include_dirs = []
        templated_subkernels = []
        assembly_calls = OrderedDict([(it, []) for it in self.supported_integral_types])
        subdomain_calls = OrderedDict([(sd, []) for sd in self.supported_subdomain_types])
        coords = None
        oriented = False
        needs_cell_sizes = False

        # Maps integral type to subdomain key
        subdomain_map = {"exterior_facet": "subdomains_exterior_facet",
                         "exterior_facet_vert": "subdomains_exterior_facet",
                         "interior_facet": "subdomains_interior_facet",
                         "interior_facet_vert": "subdomains_interior_facet"}
        for cxt_kernel in self.context_kernels:
            local_coefficients = cxt_kernel.coefficients
            it_type = cxt_kernel.original_integral_type
            exp = cxt_kernel.tensor

            if it_type not in self.supported_integral_types:
                raise ValueError("Integral type '%s' not recognized" % it_type)

            # Explicit checking of coordinates
            coordinates = cxt_kernel.tensor.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords, "Mismatching coordinates!"
            else:
                coords = coordinates

            for split_kernel in cxt_kernel.tsfc_kernels:
                kinfo = split_kernel.kinfo
                kint_type = kinfo.integral_type
                needs_cell_sizes = needs_cell_sizes or kinfo.needs_cell_sizes

                args = [c for i, _ in kinfo.coefficient_map
                        for c in self.coefficient(local_coefficients[i])]

                if kinfo.oriented:
                    args.insert(0, self.cell_orientations_sym)

                if kint_type in ["interior_facet",
                                 "exterior_facet",
                                 "interior_facet_vert",
                                 "exterior_facet_vert"]:
                    args.append(ast.FlatBlock("&%s" % self.it_sym))

                if kinfo.needs_cell_sizes:
                    args.append(self.cell_size_sym)

                # Assembly calls within the macro kernel
                call = ast.FunCall(kinfo.kernel.name,
                                   self.temps[exp],
                                   self.coord_sym,
                                   *args)

                # Subdomains only implemented for exterior facet integrals
                if kinfo.subdomain_id != "otherwise":
                    if kint_type not in subdomain_map:
                        msg = "Subdomains for integral type '%s' not implemented" % kint_type
                        raise NotImplementedError(msg)

                    sd_id = kinfo.subdomain_id
                    sd_key = subdomain_map[kint_type]
                    subdomain_calls[sd_key].append((sd_id, call))
                else:
                    assembly_calls[it_type].append(call)

                # Subkernels for local assembly (Eigen templated functions)
                from coffee.base import Node
                assert isinstance(kinfo.kernel.code, Node)
                kast = transformer.visit(kinfo.kernel.code)
                templated_subkernels.append(kast)
                include_dirs.extend(kinfo.kernel.include_dirs)
                oriented = oriented or kinfo.oriented

        # Add subdomain call to assembly dict
        assembly_calls.update(subdomain_calls)

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels
        self.include_dirs = list(set(include_dirs))
        self.oriented = oriented
        self.needs_cell_sizes = needs_cell_sizes

    @cached_property
    def coefficient_map(self):
        """Generates a mapping from a coefficient to its kernel argument
        symbol. If the coefficient is mixed, all of its split components
        will be returned.
        """
        coefficient_map = OrderedDict()
        for i, coefficient in enumerate(self.expression.coefficients()):
            if type(coefficient.ufl_element()) == MixedElement:
                csym_info = []
                for j, _ in enumerate(coefficient.split()):
                    csym_info.append(ast.Symbol("w_%d_%d" % (i, j)))
            else:
                csym_info = (ast.Symbol("w_%d" % i),)

            coefficient_map[coefficient] = tuple(csym_info)

        return coefficient_map

    def coefficient(self, coefficient):
        """Extracts the kernel arguments corresponding to a particular coefficient.
        This handles both the case when the coefficient is defined on a mixed
        or non-mixed function space.
        """
        return self.coefficient_map[coefficient]

    @cached_property
    def context_kernels(self):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """

        cxt_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters,
                                          coffee=True)
                    for i, expr in enumerate(self.temps)]

        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    @property
    def integral_type(self):
        """Returns the integral type associated with a Slate kernel."""
        return "cell"

    @cached_property
    def needs_cell_facets(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require looping over cell facets. If any are found, this function
        returns `True` and `False` otherwise.
        """
        cell_facet_types = ["interior_facet",
                            "exterior_facet",
                            "interior_facet_vert",
                            "exterior_facet_vert"]
        return any(cxt_k.original_integral_type in cell_facet_types
                   for cxt_k in self.context_kernels)

    @cached_property
    def needs_mesh_layers(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require mesh level information (extrusion measures). If any are
        found, this function returns `True` and `False` otherwise.
        """
        mesh_layer_types = ["interior_facet_horiz_top",
                            "interior_facet_horiz_bottom",
                            "exterior_facet_bottom",
                            "exterior_facet_top"]
        return any(cxt_k.original_integral_type in mesh_layer_types
                   for cxt_k in self.context_kernels)


class LocalLoopyKernelBuilder(object):

    coordinates_arg_name = "coords"
    cell_facets_arg_name = "cell_facets"
    local_facet_array_arg_name = "facet_array"
    layer_arg_name = "layer"
    layer_count_name = "layer_count"
    cell_sizes_arg_name = "cell_sizes"
    cell_orientations_arg_name = "cell_orientations"

    # Supported integral types
    supported_integral_types = [
        "cell",
        "interior_facet",
        "exterior_facet",
        # The "interior_facet_horiz" measure is separated into two parts:
        # "top" and "bottom"
        "interior_facet_horiz_top",
        "interior_facet_horiz_bottom",
        "interior_facet_vert",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "exterior_facet_vert"
    ]

    # Supported subdomain types
    supported_subdomain_types = ["subdomains_exterior_facet",
                                 "subdomains_interior_facet"]

    def __init__(self, expression, tsfc_parameters=None, slate_loopy_name=None):
        """Constructor for the LocalGEMKernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
            TSFC when constructing subkernels associated with the expression.
        """

        assert isinstance(expression, slate.TensorBase)

        self.slate_loopy_name = slate_loopy_name
        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.bag = SlateWrapperBag({})
        self.matfree_solve_knls = []

    def tsfc_cxt_kernels(self, terminal):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """

        return compile_terminal_form(terminal, prefix="subkernel%d_" % knl_counter(),
                                     tsfc_parameters=self.tsfc_parameters, coffee=False)

    def shape(self, tensor):
        """ A helper method to retrieve tensor shape information.
        In particular needed for the right shape of scalar tensors.
        """
        if tensor.shape == ():
            return (1, )  # scalar tensor
        else:
            return tensor.shape

    def extent(self, coefficient):
        """ Calculation of the range of a coefficient."""
        element = coefficient.ufl_element()
        if element.family() == "Real":
            return (coefficient.dat.cdim, )
        else:
            return (create_element(element).space_dimension(), )

    def generate_lhs(self, tensor, temp):
        """ Generation of an lhs for the loopy kernel,
            which contains the TSFC assembly of the tensor.
        """
        idx = self.bag.index_creator(self.shape(tensor))
        lhs = pym.Subscript(temp, idx)
        return SubArrayRef(idx, lhs)

    def collect_tsfc_kernel_data(self, mesh, tsfc_coefficients, kinfo):
        """ Collect the kernel data aka the parameters fed into the subkernel,
            that are coordinates, orientations, cell sizes and cofficients.
        """

        kernel_data = [(mesh.coordinates, self.coordinates_arg_name)]

        if kinfo.oriented:
            self.bag.needs_cell_orientations = True
            kernel_data.append((mesh.cell_orientations(), self.cell_orientations_arg_name))

        if kinfo.needs_cell_sizes:
            self.bag.needs_cell_sizes = True
            kernel_data.append((mesh.cell_sizes, self.cell_sizes_arg_name))

        # Append original coefficients from the expression
        if self.bag.coefficients:
            kernel_data.extend(self.collect_tsfc_coefficients(tsfc_coefficients,
                                                              kinfo,
                                                              self.bag.coefficients))

        # Append artificial coefficients for action
        if self.bag.action_coefficients:
            kernel_data.extend(self.collect_tsfc_coefficients(tsfc_coefficients,
                                                              kinfo,
                                                              self.bag.action_coefficients,
                                                              True))
        return kernel_data

    def collect_tsfc_coefficients(self, tsfc_coefficients, kinfo, wrapper_coefficients, action=False):
        kernel_data = []
        # Pick the coefficients associated with a Tensor()/TSFC kernel
        tsfc_coefficients = [tsfc_coefficients[i] for i, _ in kinfo.coefficient_map]
        for c, cinfo in wrapper_coefficients.items():
            # All artificial coefficients, coming from actions, and
            # all original coefficients, which are also present in the TSFC kernel,
            # are relevant data for the kernel
            if c in tsfc_coefficients or action:
                if isinstance(cinfo, tuple):
                    # info for coefficients on non-mixed spaces is a tuple
                    kernel_data.extend([(c, cinfo[0])])
                else:
                    # info for coefficients on mixed spaces is a dict
                    for c_, info in cinfo.items():
                        kernel_data.extend([((c, c_), info[0])])
        return kernel_data

    def loopify_tsfc_kernel_data(self, kernel_data):
        """ This method generates loopy arguments from the kernel data,
            which are then fed to the TSFC loopy kernel. The arguments
            are arrays and have to be fed element by element to loopy
            aka they have to be subarrayrefed.
        """
        arguments = []
        offset = 0
        last_mixed_c = None
        for c, info in kernel_data:
            if isinstance(info, tuple):
                name, shape = info
                shape = shape if shape else (1,)
            else:
                name = info

            # We need to treat mixed coefficients separately for kernels generated with the new matrix-free infrastructure
            # since on ufl a level we have replaced an argument in the form with a coefficient.
            # This extra coefficient needs to be split into its parts when passed into kernels generated by the form compiler TSFC.
            # With the new code bits in this function we e.g. generate an action kernel
            # subkernel3_cell_to__cell_integral_otherwise(&(T8_x[0]), &(coords[0]), &(x[0]), &(x[3]));
            # rather than
            # subkernel3_cell_to__cell_integral_otherwise(&(T8_x[0]), &(coords[0]), &(x[0]), &(x[0]));
            # FIXME probably we can do this a bit nicer

            if isinstance(c, tuple):  # then the coeff is coming from a mixed background
                mixed_c, split_c = c
                if last_mixed_c != mixed_c:
                    # reset offset when all splits of one mixed coefficient have been dealt with
                    # and a new mixed coefficient in kernel data is dealt with
                    # NOTE this depends on all split coefficients being appended in order
                    # (which is the case but noteworthy nevertheless)
                    offset = 0
                    last_mixed_c = None

                shp2 = self.extent(split_c)
                shp, = shp2 if shp2 else (1,)
                idx = self.bag.index_creator((shp,))

                # We need to subarrayref into part of the temorary
                # to get the correct part of the mixed coefficient
                offset_index = (pym.Sum((offset, idx[0])),)
                c = pym.Subscript(pym.Variable(name), offset_index)
                arguments.append(SubArrayRef(idx, c))

                # set offset for the next split of the mixed coefficient
                offset += shp
                last_mixed_c = mixed_c
            else:
                extent = self.extent(c)
                idx = self.bag.index_creator(extent)
                arguments.append(SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx)))
        return arguments

    def layer_integral_predicates(self, tensor, integral_type):
        self.bag.needs_mesh_layers = True
        layer = pym.Variable(self.layer_arg_name)

        # TODO: Variable layers
        nlayer = pym.Variable(self.layer_count_name)
        which = {"interior_facet_horiz_top": pym.Comparison(layer, "<", nlayer[0]),
                 "interior_facet_horiz_bottom": pym.Comparison(layer, ">", 0),
                 "exterior_facet_top": pym.Comparison(layer, "==", nlayer[0]),
                 "exterior_facet_bottom": pym.Comparison(layer, "==", 0)}[integral_type]

        return [which]

    def facet_integral_predicates(self, mesh, integral_type, kinfo):
        self.bag.needs_cell_facets = True
        # Number of recerence cell facets
        if mesh.cell_set._extruded:
            self.bag.num_facets = mesh._base_mesh.ufl_cell().num_facets()
        else:
            self.bag.num_facets = mesh.ufl_cell().num_facets()

        # Index for loop over cell faces of reference cell
        fidx = self.bag.index_creator((self.bag.num_facets,))

        # Cell is interior or exterior
        select = 1 if integral_type.startswith("interior_facet") else 0

        i = self.bag.index_creator((1,))
        predicates = [pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg_name), (fidx[0], 0)), "==", select)]

        # TODO subdomain boundary integrals, this does the wrong thing for integrals like f*ds + g*ds(1)
        # "otherwise" is treated incorrectly as "everywhere"
        # However, this replicates an existing slate bug.
        if kinfo.subdomain_id != "otherwise":
            predicates.append(pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg_name), (fidx[0], 1)), "==", kinfo.subdomain_id))

        # Additional facet array argument to be fed into tsfc loopy kernel
        subscript = pym.Subscript(pym.Variable(self.local_facet_array_arg_name),
                                  (pym.Sum((i[0], fidx[0]))))
        facet_arg = SubArrayRef(i, subscript)

        return predicates, fidx, facet_arg

    # TODO: is this ugly?
    def is_integral_type(self, integral_type, type):
        cell_integral = ["cell"]
        facet_integral = ["interior_facet",
                          "interior_facet_vert",
                          "exterior_facet",
                          "exterior_facet_vert"]
        layer_integral = ["interior_facet_horiz_top",
                          "interior_facet_horiz_bottom",
                          "exterior_facet_top",
                          "exterior_facet_bottom"]
        if ((integral_type in cell_integral and type == "cell_integral")
           or (integral_type in facet_integral and type == "facet_integral")
           or (integral_type in layer_integral and type == "layer_integral")):
            return True
        else:
            return False

    def collect_coefficients(self, expr=None, names=None, artificial=True):
        """ Saves all coefficients of self.expression, where non mixed coefficient
            are of dict of form {coff: (name, extent)} and mixed coefficient are
            double dict of form {mixed_coeff: {coeff_per_space: (name,extent)}}.
            The coefficients are seperated into original coefficients coming from
            the expression and artificial ones used for actions.
        """
        expr = expr if expr else self.expression
        coeffs = expr.coefficients(artificial=artificial)  #FXIME form updating to master
        coeff_dict = OrderedDict()
        new_coeff_dict = OrderedDict()

        # TODO is there are better way to do this?
        for i, (c, split_map) in enumerate(self.expression.coeff_map):
            new = False
            try:
                # check if the coefficient is in names,
                # if yes it will be replaced later
                prefix = names[c]
                new = True
            except (KeyError, TypeError):
                # if coefficient is not in names it is not an
                # an action coefficient so we can use usual naming conventions
                if not new:
                    prefix = "w_{}".format(i)
            element = c.ufl_element()
            # collect information about the coefficient in particular name and extent
            if type(coeff.ufl_element()) == MixedElement:
                info = OrderedDict()
                splits = coeff.split()
                info = OrderedDict({splits[j]: (f"{prefix}_{j}", self.extent(splits[j])) for j in split_map})
            else:
                info = (prefix, self.extent(coeff))
            if new:
                new_coeff_dict[c] = info
            else:
                coeff_dict[c] = info
        return coeff_dict, new_coeff_dict

    def initialise_terminals(self, var2tensor, coefficients):
        """ Initilisation of the variables in which coefficients
            and the Tensors coming from TSFC are saved.
            For matrix-free kernels Actions are initialised too.

            :arg var2terminal: dictionary that maps gem Variables to Slate tensors
        """

        from gem import Variable as gVar, Action
        var2terminal = dict(filter(lambda elem: isinstance(elem[0], gVar) or isinstance(elem[0], Action), var2tensor.items()))
        tensor2temp = OrderedDict()
        inits = []
        for gem_tensor, slate_tensor in var2terminal.items():
            (_, dtype), = assign_dtypes([gem_tensor], self.tsfc_parameters["scalar_type"])
            loopy_tensor = loopy.TemporaryVariable(gem_tensor.name,
                                                   dtype=dtype,
                                                   shape=gem_tensor.shape,
                                                   address_space=loopy.AddressSpace.LOCAL,
                                                   target=loopy.CTarget())
            tensor2temp[slate_tensor] = loopy_tensor

            if not slate_tensor.assembled:
                indices = self.bag.index_creator(self.shape(slate_tensor))
                inames = {var.name for var in indices}
                var = pym.Subscript(pym.Variable(loopy_tensor.name), indices)
                inits.append(loopy.Assignment(var, "0.", id="init_" + gem_tensor.name,
                                              within_inames=frozenset(inames),
                                              within_inames_is_final=True))
            else:
                potentially_mixed_f = slate_tensor.form if isinstance(slate_tensor.form, tuple) else (slate_tensor.form,)
                coeff_dict = tuple(coefficients[c] for c in potentially_mixed_f)
                offset = 0
                ismixed = tuple((type(c.ufl_element()) == MixedElement) for c in potentially_mixed_f)
                # Fetch the coefficient name corresponding to this assembled vector
                names = []
                for (im, c) in zip(ismixed, coeff_dict):
                    if im:
                        # For block assembled vectors we need to pick the right name
                        # corresponding to the split function of the block assembled vector
                        block_function, = slate_tensor.slate_coefficients()
                        filter_f = lambda name_and_extent: (not isinstance(block_function, slate.BlockFunction)
                                                            or name_and_extent[0] in block_function.split_function)
                        names += list(name for (_, (name, _)) in tuple(filter(filter_f, c.items())))
                    else:
                        names += [c[0]]
                # Mixed coefficients come as seperate parameter (one per space)
                for i, shp in enumerate(*slate_tensor.shapes.values()):
                    indices = self.bag.index_creator((shp,))
                    inames = {var.name for var in indices}
                    offset_index = (pym.Sum((offset, indices[0])),)
                    name = names[i] if ismixed else names
                    var = pym.Subscript(pym.Variable(loopy_tensor.name), offset_index)
                    c = pym.Subscript(pym.Variable(name), indices)
                    inits.append(loopy.Assignment(var, c, id="init_" + gem_tensor.name + "_" + str(i),
                                                  within_inames=frozenset(inames),
                                                  within_inames_is_final=True))
                    offset += shp

        return inits, tensor2temp

    def slate_call(self, kernel, temporaries):
        output_var = pym.Variable(kernel.args[0].name)
        # Slate kernel call
        reads = [output_var]
        for t in temporaries:
            shape = t.shape
            name = t.name
            idx = self.bag.index_creator(shape)
            reads.append(SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx)))
        call = pym.Call(pym.Variable(kernel.name), tuple(reads))
        output_var = pym.Variable(kernel.args[0].name)
        slate_kernel_call_output = self.generate_lhs(self.expression, output_var)
        insn = loopy.CallInstruction((slate_kernel_call_output,), call, id="slate_kernel_call")
        return insn

    def generate_matfsolve_call(self, ctx, insn, expr):
        """
            Matrix-free solve. Currently implemented as CG.
        """
        knl_no = knl_counter()
        name = "mtf_solve_%d" % knl_no
        shape = expr.shape
        dtype = self.tsfc_parameters["scalar_type"]

        # Generate the arguments for the kernel from the loopy expression
        args, reads, output_arg = self.generate_kernel_args_and_call_reads(expr, insn, dtype)

        # Map from local kernel arg name to global arg name
        str2name = {}
        local_names = ["A", "output", "b"]
        coeff_arg = None
        for c, arg in enumerate(args):
            if (arg.name in [self.coordinates_arg, self.cell_facets_arg, self.local_facet_array_arg,
                             self.cell_size_arg, self.cell_orientations_arg]
                or arg.name in [coeff[0] if isinstance(coeff, tuple)
                                else coeff for coeff in self.bag.coefficients.values()]):
                local_names.insert(c, arg.name)
            str2name[local_names[c]] = arg.name
            if local_names[c] == "b":
                coeff_arg = arg

        # rename x and p in case they are already arguments
        str2name["x"] = "x"
        str2name["p"] = "p"
        for arg in args:
            str2name["x"] = "x" + str(knl_no) if arg.name == "x" else str2name["x"]
            str2name["p"] = "p" + str(knl_no) if arg.name == "p" else str2name["p"]

        # name of the lhs to the action call inside the matfree solve kernel
        child1, _ = expr.children
        A_on_x_name = ctx.gem_to_pymbolic[child1].name+"_x" if not hasattr(expr.Aonx, "name") else expr.Aonx.name
        A_on_p_name = ctx.gem_to_pymbolic[child1].name+"_p" if not hasattr(expr.Aonp, "name") else expr.Aonp.name
        str2name.update({"A_on_x": A_on_x_name, "A_on_p": A_on_p_name})

        # setup the stop criterions
        str2name.update({"stop_criterion_id": "cond", "stop_criterion_dep": "rkp1_normk"})
        stop_criterion = self.generate_code_for_stop_criterion("rkp1_norm",
                                                               1.e-16,
                                                               str2name["stop_criterion_dep"],
                                                               str2name["stop_criterion_id"])
        str2name.update({"preconverged_criterion_id": "projis0", "preconverged_criterion_dep": "projector"})
        corner_case = self.generate_code_for_converged_pre_iteration(str2name["preconverged_criterion_dep"],
                                                                     str2name["preconverged_criterion_id"])

        # NOTE The last line in the loop to convergence is another WORKAROUND
        # bc the initialisation of A_on_p in the action call does not get inlined properly either
        knl = loopy.make_function(
            """{[i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7,i_8,i_9,i_10,i_11,i_12]:
                 0<=i_0,i_1,i_2,i_3,i_4,i_5,i_7,i_8,i_9,i_10,i_11,i_12<n
                 and 0<=i_6<=n}""",
            ["""{x}[i_0] = -{b}[i_0] {{id=x0}}
                {A_on_x}[:] = action_A({A}[:,:], {x}[:]) {{dep=x0, id=Aonx}}
                <> r[i_3] = {A_on_x}[i_3]-{b}[i_3] {{dep=Aonx, id=residual0}}
                {p}[i_4] = -r[i_4] {{dep=residual0, id=projector0}}
                <> rk_norm = 0. {{dep=projector0, id=rk_norm0}}
                rk_norm = rk_norm + r[i_5]*r[i_5] {{dep=projector0, id=rk_norm1}}
                for i_6
                    {A_on_p}[:] = action_A_on_p({A}[:,:], {p}[:]) {{dep=Aonp0, id=Aonp, inames=i_6}}
                    <> p_on_Ap = 0. {{dep=Aonp, id=ponAp0}}
                    p_on_Ap = p_on_Ap + {p}[i_2]*{A_on_p}[i_2] {{dep=ponAp0, id=ponAp}}
                    <> projector_is_zero = abs(p_on_Ap) < 1.e-16 {{id={preconverged_criterion_dep}, dep=ponAp}}
             """.format(**str2name),
             corner_case,
             """    <> alpha = rk_norm / p_on_Ap {{dep={preconverged_criterion_id}, id=alpha}}
                    {x}[i_7] = {x}[i_7] + alpha*{p}[i_7] {{dep=ponAp, id=xk}}
                    r[i_8] = r[i_8] + alpha*{A_on_p}[i_8] {{dep=xk,id=rk}}
                    <> rkp1_norm = 0. {{dep=rk, id=rkp1_norm0}}
                    rkp1_norm = rkp1_norm + r[i_9]*r[i_9] {{dep=rkp1_norm0, id={stop_criterion_dep}}}
             """.format(**str2name),
             stop_criterion,
             """    <> beta = rkp1_norm / rk_norm {{dep={stop_criterion_id}, id=beta}}
                    rk_norm = rkp1_norm {{dep=beta, id=rk_normk}}
                    {p}[i_10] = beta * {p}[i_10] - r[i_10] {{dep=rk_normk, id=projectork}}
                    {A_on_p}[i_12] = 0. {{dep=projectork, id=Aonp0, inames=i_6}}
                end
                {output}[i_11] = {x}[i_11] {{dep=Aonp0, id=out}}
             """.format(**str2name)],
            [*args,
             loopy.TemporaryVariable(str2name["x"], dtype, shape=shape, address_space=loopy.AddressSpace.LOCAL, target=loopy.CTarget()),
             loopy.TemporaryVariable(A_on_x_name, dtype, shape=shape, address_space=loopy.AddressSpace.LOCAL),
             loopy.TemporaryVariable(A_on_p_name, dtype, shape=shape, address_space=loopy.AddressSpace.LOCAL),
             loopy.TemporaryVariable(str2name["p"], dtype, shape=shape, address_space=loopy.AddressSpace.LOCAL)],
            target=loopy.CTarget(),
            name=name,
            lang_version=(2018, 2),
            silenced_warnings=["single_writer_after_creation", "unused_inames"])

        # set the length of the indices to the size of the temporaries
        knl = loopy.fix_parameters(knl, n=shape[0])

        # update gem to pym mapping
        # by linking the actions of the matrix-free solve kernel
        # to the their pymbolic variables
        ctx.gem_to_pymbolic[expr.Aonx] = pym.Variable(knl.callables_table[name].subkernel.id_to_insn["Aonx"].assignees[0].subscript.aggregate.name)
        ctx.gem_to_pymbolic[expr.Aonp] = pym.Variable(knl.callables_table[name].subkernel.id_to_insn["Aonp"].assignees[0].subscript.aggregate.name)

        # the expression which call the knl for the matfree solve kernel
        call = insn.copy(expression=pym.Call(pym.Variable(name), reads))

        self.matfree_solve_knls.append(knl)
        return call, (name, knl), output_arg, ctx, coeff_arg

    def generate_code_for_converged_pre_iteration(self, dep, id):
        if configuration["simd_width"]:
            assert "not vectorised yet"
        return loopy.CInstruction("",
                                  "if (projector_is_zero) break;",
                                  depends_on=dep,
                                  id=id)

    def generate_code_for_stop_criterion(self, var_name, stop_value, dep, id):
        """ This method is workaround need since Loo.py does not support while loops yet.
            FIXME whenever while loops become available

            The workaround uses a Loo.py CInstruction. The Loo.py Cinstruction allows to write C code
            so that the code (defined via its second argument) will appear unaltered in the final
            produced code for the kernel. Meaning, there are no transformations happening on this
            bit of code.
            First example where this becomes a problem is in a kernel containing the Cinstruction,
            which gets inlined in another kernel. In that case the variables in the instruction
            are not renamed properly in the inlining process.
            Another example is, that the variable in the code of the Cinstruction does not get
            vectorised when prompted.

            Inlining and vectorisation are made available through this ugly bit of code.
        """
        condition = " < " + str(stop_value)
        if configuration["simd_width"]:
            # vectorisation of the stop criterion
            variable = var_name + "[" + str(0) + "]" + condition
            for i in range(int(configuration["simd_width"])-1):
                variable += "&& " + var_name + "["+str(i+1)+"]" + condition
        else:
            variable = var_name + condition
        return loopy.CInstruction("",
                                  "if (" + variable + ") break;",
                                  read_variables=[var_name],
                                  depends_on=dep,
                                  id=id)

    def generate_kernel_args_and_call_reads(self, expr, insn, dtype):
        """A function which is used for generating the arguments to the kernel and the read variables
           for the call to that kernel of the matrix-free solve."""
        child1, child2 = expr.children
        reads1, reads2 = insn.expression.parameters

        # Generate kernel args
        arg1 = loopy.GlobalArg(reads1.subscript.aggregate.name, dtype, shape=child1.shape, is_output=False, is_input=True,
                               target=loopy.CTarget(), dim_tags=None, strides=loopy.auto, order='C')
        arg2 = loopy.GlobalArg(reads2.subscript.aggregate.name, dtype, shape=child2.shape, is_output=False, is_input=True,
                               target=loopy.CTarget(), dim_tags=None, strides=loopy.auto, order='C')
        output_arg = loopy.GlobalArg(insn.assignee_name, dtype, shape=expr.shape, is_output=True, is_input=True,
                                     target=loopy.CTarget(), dim_tags=None, strides=loopy.auto, order='C')

        args = self.generate_wrapper_kernel_args()
        args.append(arg2)
        args.insert(0, output_arg)
        args.insert(0, arg1)

        # Generate call parameters
        reads = []
        for arg in args:
            var_reads = pym.Variable(arg.name)
            idx_reads = self.bag.index_creator(arg.shape)
            reads.append(SubArrayRef(idx_reads, pym.Subscript(var_reads, idx_reads)))
        return args, reads, output_arg

    def generate_wrapper_kernel_args(self, temporaries={}):
        args = []
        tmp_args = []
        # FIXME if we really need the dimtags and so on
        # maybe we should make a function for generating global args
        coords_extent = self.extent(self.expression.ufl_domain().coordinates)
        coords_loopy_arg = loopy.GlobalArg(self.coordinates_arg_name, shape=coords_extent,
                                           dtype=self.tsfc_parameters["scalar_type"],
                                           dim_tags=None, strides=loopy.auto, order="C",
                                           target=loopy.CTarget(), is_input=True, is_output=False))
        args.append(kernel_args.CoordinatesKernelArg(coords_loopy_arg))

        if self.bag.needs_cell_orientations:
            ori_extent = self.extent(self.expression.ufl_domain().cell_orientations())
            ori_loopy_arg = loopy.GlobalArg(self.cell_orientations_arg_name,
                                            shape=ori_extent,
                                            dtype=self.tsfc_parameters["scalar_type"],
                                            target=loopy.CTarget(),
                                            is_input=True, is_output=False,
                                            dim_tags=None, strides=loopy.auto, order="C"))
            args.append(kernel_args.CellOrientationsKernelArg(ori_loopy_arg))

        if self.bag.needs_cell_sizes:
            siz_extent = self.extent(self.expression.ufl_domain().cell_sizes)
            siz_loopy_arg = loopy.GlobalArg(self.cell_sizes_arg_name, shape=siz_extent,
                                            dtype=self.tsfc_parameters["scalar_type"],
                                            is_input=True, is_output=False,
                                            dim_tags=None, strides=loopy.auto, order="C")
            args.append(kernel_args.CellSizesKernelArg(siz_loopy_arg))

        for coeff in self.bag.coefficients.values():
            if isinstance(coeff, OrderedDict):
                for name, extent in coeff.values():
                    coeff_loopy_arg = loopy.GlobalArg(name, shape=extent,
                                                      dtype=self.tsfc_parameters["scalar_type"],   
                                                      target=loopy.CTarget(),
                                                      is_input=True, is_output=False,
                                                      dim_tags=None, strides=loopy.auto, order="C")
                    args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))
            else:
                name, extent = coeff
                coeff_loopy_arg = loopy.GlobalArg(name, shape=extent,
                                                  dtype=self.tsfc_parameters["scalar_type"],
                                                  target=loopy.CTarget(),
                                                  is_input=True, is_output=False,
                                                  dim_tags=None, strides=loopy.auto, order="C")
                args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))

        if self.bag.needs_cell_facets:
            # Arg for is exterior (==0)/interior (==1) facet or not
            facet_loopy_arg = loopy.GlobalArg(self.cell_facets_arg_name,
                                              shape=(self.num_facets, 2),
                                              dtype=np.int8, is_input=True, is_output=False,
                                              target=loopy.CTarget(),
                                              dim_tags=None, strides=loopy.auto, order="C"))
            args.append(CellFacetKernelArg(facet_loopy_arg))

            tmp_args.append(loopy.TemporaryVariable(self.local_facet_array_arg_name,
                                                    shape=(self.bag.num_facets,),
                                                    dtype=np.uint32,
                                                    address_space=loopy.AddressSpace.LOCAL,
                                                    read_only=True,
                                                    initializer=np.arange(self.bag.num_facets, dtype=np.uint32),
                                                    target=loopy.CTarget(),
                                                    dim_tags=None, strides=loopy.auto, order="C"))

        if self.bag.needs_mesh_layers:
            layer_loopy_arg = loopy.GlobalArg(self.layer_count_name, shape=(1,),
                                              dtype=np.int32, is_input=True, is_output=False,
                                              target=loopy.CTarget(),
                                              dim_tags=None, strides=loopy.auto, order="C")
            args.append(LayerCountKernelArg(layer_loopy_arg))

            tmp_args.append(loopy.ValueArg(self.layer_arg_name, dtype=np.int32))

        for tensor_temp in temporaries:
            args.append(tensor_temp)

        return args, tmp_args

    def generate_tsfc_calls(self, terminal, loopy_tensor):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC. This function also collects any
        information regarding orientations and extra include directories.
        """
        cxt_kernels = self.tsfc_cxt_kernels(terminal)

        for cxt_kernel in cxt_kernels:
            for tsfc_kernel in cxt_kernel.tsfc_kernels:
                integral_type = cxt_kernel.original_integral_type
                slate_tensor = cxt_kernel.tensor
                mesh = slate_tensor.ufl_domain()
                kinfo = tsfc_kernel.kinfo
                reads = []
                inames_dep = []

                if integral_type not in self.supported_integral_types:
                    raise ValueError("Integral type '%s' not recognized" % integral_type)

                # Prepare lhs and args for call to tsfc kernel
                output_var = pym.Variable(loopy_tensor.name)
                output = self.generate_lhs(slate_tensor, output_var)
                kernel_data = self.collect_tsfc_kernel_data(mesh, cxt_kernel.coefficients, kinfo)
                reads.append(output)
                reads.extend(self.loopify_tsfc_kernel_data(kernel_data))

                # Generate predicates for different integral types
                if self.is_integral_type(integral_type, "cell_integral"):
                    predicates = None
                    if kinfo.subdomain_id != "otherwise":
                        raise NotImplementedError("No subdomain markers for cells yet")
                elif self.is_integral_type(integral_type, "facet_integral"):
                    predicates, fidx, facet_arg = self.facet_integral_predicates(mesh, integral_type, kinfo)
                    reads.append(facet_arg)
                    inames_dep.append(fidx[0].name)
                elif self.is_integral_type(integral_type, "layer_integral"):
                    predicates = self.layer_integral_predicates(slate_tensor, integral_type)
                else:
                    raise ValueError("Unhandled integral type {}".format(integral_type))

                # TSFC kernel call
                key = self.bag.call_name_generator(integral_type)
                call = pym.Call(pym.Variable(kinfo.kernel.name), tuple(reads))
                insn = loopy.CallInstruction((output,), call,
                                             within_inames=frozenset(inames_dep),
                                             predicates=predicates, id=key)
                event, = kinfo.events
                yield insn, {kinfo.kernel.name: kinfo.kernel.code}, event

        # tsfc yields no kernels if they'd reduce to T0 = 0
        if not cxt_kernels:
            yield (None, None)


class SlateWrapperBag(object):

    def __init__(self, coeffs, action_coeffs={}, name=""):
        self.coefficients = coeffs
        self.action_coefficients = action_coeffs
        self.needs_cell_orientations = False
        self.needs_cell_sizes = False
        self.needs_cell_facets = False
        self.needs_mesh_layers = False
        self.num_facets = None
        self.call_name_generator = UniqueNameGenerator()
        self.index_creator = IndexCreator()
        self.name = name

    def copy_extra_args(self, other):
        self.coefficients = other.coefficients
        if not self.needs_cell_orientations:
            self.needs_cell_orientations = other.needs_cell_orientations
        if not self.needs_cell_sizes:
            self.needs_cell_sizes = other.needs_cell_sizes
        if not self.needs_cell_facets:
            self.needs_cell_facets = other.needs_cell_facets
            self.num_facets = other.num_facets
        if not self.needs_mesh_layers:
            self.needs_mesh_layers = other.needs_mesh_layers

    def copy_coefficients(self, coeffs=None, action_coeffs=None):
        if coeffs:
            self.coefficients = coeffs
        if action_coeffs:
            self.action_coefficients = action_coeffs


class IndexCreator(object):
    def __init__(self):
        self.inames = OrderedDict()  # pym variable -> extent
        self.namer = UniqueNameGenerator()

    def __call__(self, extents):
        """Create new indices with specified extents.

        :arg extents. :class:`tuple` containting :class:`tuple` for extents of mixed tensors
            and :class:`int` for extents non-mixed tensor
        :returns: tuple of pymbolic Variable objects representing indices, contains tuples
            of Variables for mixed tensors
            and Variables for non-mixed tensors, where each Variable represents one extent."""

        # Indices for scalar tensors
        extents += (1, ) if len(extents) == 0 else ()

        # Stacked tuple = mixed tensor
        # -> loop over ext to generate idxs per block
        indices = []
        if isinstance(extents[0], tuple):
            for ext_per_block in extents:
                idxs = self._create_indices(ext_per_block)
                indices.append(idxs)
            return tuple(indices)
        # Non-mixed tensors
        else:
            return self._create_indices(extents)

    def _create_indices(self, extents):
        """Create new indices with specified extents.

        :arg extents. :class:`tuple` or :class:`int` for extent of each index
        :returns: tuple of pymbolic Variable objects representing
            indices, one for each extent."""
        indices = []
        for ext in extents:
            name = self.namer()
            indices.append(pym.Variable(name))
            if name not in self.inames.keys():
                self.inames[name] = int(ext)
            else:
                if self.inames[name] != ext:
                    raise KeyError("""An index that has already been generated is attempted to be recreated with a new extent.
                                    This should not happen. Make sure your index naming is unique.""")
        return tuple(indices)

    @property
    def domains(self):
        """ISL domains for the currently known indices."""
        return create_domains(self.inames.items())
