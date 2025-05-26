
from __future__ import division

import numba as nb
import numpy as np


'''
***************** util.py *****************
'''

funcs_common = 'first last len mean var std allnan anynan max min argmax argmin sumofsquares cumsum cumprod cummax cummin'.split()
funcs_no_separate_nan = frozenset(['sort', 'rsort', 'array', 'allnan', 'anynan'])


_alias_str = {
    'or': 'any',
    'and': 'all',
    'add': 'sum',
    'count': 'len',
    'plus': 'sum',
    'multiply': 'prod',
    'product': 'prod',
    'times': 'prod',
    'amax': 'max',
    'maximum': 'max',
    'amin': 'min',
    'minimum': 'min',
    'split': 'array',
    'splice': 'array',
    'sorted': 'sort',
    'asort': 'sort',
    'asorted': 'sort',
    'rsorted': 'sort',
    'dsort': 'sort',
    'dsorted': 'rsort',
}

_alias_builtin = {
    all: 'all',
    any: 'any',
    len: 'len',
    max: 'max',
    min: 'min',
    sum: 'sum',
    sorted: 'sort',
    slice: 'array',
    list: 'array',
}


def get_aliasing(*extra):
    """The assembles the dict mapping strings and functions to the list of
    supported function names:
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'
    This funciton should only be called during import.
    """
    alias = dict((k, k) for k in funcs_common)
    alias.update(_alias_str)
    alias.update((fn, fn) for fn in _alias_builtin.values())
    alias.update(_alias_builtin)
    for d in extra:
        alias.update(d)
    alias.update((k, k) for k in set(alias.values()))
    for key in set(alias.values()):
        if key not in funcs_no_separate_nan:
            key = 'nan' + key
            alias[key] = key
    return alias


aliasing = get_aliasing()


def get_func(func, aliasing, implementations):
    """ Return the key of a found implementation or the func itself """
    try:
        func_str = aliasing[func]
    except KeyError:
        if callable(func):
            return func
    else:
        if func_str in implementations:
            return func_str
        if func_str.startswith('nan') and \
                func_str[3:] in funcs_no_separate_nan:
            raise ValueError("%s does not have a nan-version".format(func_str[3:]))
        else:
            raise NotImplementedError("No such function available")
    raise ValueError("func {} is neither a valid function string nor a "
                     "callable object".format(func))


def check_boolean(x):
    if x not in (0, 1):
        raise ValueError("Value not boolean")


try:
    basestring

    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)


'''
***************** numpy_util.py *****************
'''


_alias_numpy = {
    np.add: 'sum',
    np.sum: 'sum',
    np.any: 'any',
    np.all: 'all',
    np.multiply: 'prod',
    np.prod: 'prod',
    np.amin: 'min',
    np.min: 'min',
    np.minimum: 'min',
    np.amax: 'max',
    np.max: 'max',
    np.maximum: 'max',
    np.argmax: 'argmax',
    np.argmin: 'argmin',
    np.mean: 'mean',
    np.std: 'std',
    np.var: 'var',
    np.array: 'array',
    np.asarray: 'array',
    np.sort: 'sort',
    np.nansum: 'nansum',
    np.nanprod: 'nanprod',
    np.nanmean: 'nanmean',
    np.nanvar: 'nanvar',
    np.nanmax: 'nanmax',
    np.nanmin: 'nanmin',
    np.nanstd: 'nanstd',
    np.nanargmax: 'nanargmax',
    np.nanargmin: 'nanargmin',
    np.cumsum: 'cumsum',
    np.cumprod: 'cumprod',
}

aliasing = get_aliasing(_alias_numpy)

_next_int_dtype = dict(
    bool=np.int8,
    uint8=np.int16,
    int8=np.int16,
    uint16=np.int32,
    int16=np.int32,
    uint32=np.int64,
    int32=np.int64
)

_next_float_dtype = dict(
    float16=np.float32,
    float32=np.float64,
    float64=np.complex64,
    complex64=np.complex128
)


def minimum_dtype(x, dtype=np.bool_):
    """returns the "most basic" dtype which represents `x` properly, which
    provides at least the same value range as the specified dtype."""

    def check_type(x, dtype):
        try:
            converted = dtype.type(x)
        except (ValueError, OverflowError):
            return False
        return converted == x or np.isnan(x)

    def type_loop(x, dtype, dtype_dict, default=None):
        while True:
            try:
                dtype = np.dtype(dtype_dict[dtype.name])
                if check_type(x, dtype):
                    return np.dtype(dtype)
            except KeyError:
                if default is not None:
                    return np.dtype(default)
                raise ValueError("Can not determine dtype of %r" % x)

    dtype = np.dtype(dtype)
    if check_type(x, dtype):
        return dtype

    if np.issubdtype(dtype, np.inexact):
        return type_loop(x, dtype, _next_float_dtype)
    else:
        return type_loop(x, dtype, _next_int_dtype, default=np.float32)


def minimum_dtype_scalar(x, dtype, a):
    if dtype is None:
        dtype = np.dtype(type(a)) if isinstance(a, (int, float)) else a.dtype
    return minimum_dtype(x, dtype)


_forced_types = {
    'array': object,
    'all': bool,
    'any': bool,
    'nanall': bool,
    'nanany': bool,
    'len': np.int64,
    'nanlen': np.int64,
    'allnan': bool,
    'anynan': bool,
    'argmax': np.int64,
    'argmin': np.int64,
    'nanargmin': np.int64,
    'nanargmax': np.int64,
}
_forced_float_types = {'mean', 'var', 'std', 'nanmean', 'nanvar', 'nanstd'}
_forced_same_type = {'min', 'max', 'first', 'last', 'nanmin', 'nanmax',
                     'nanfirst', 'nanlast'}


def check_dtype(dtype, func_str, a, n):
    if np.isscalar(a) or not a.shape:
        if func_str not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', "
                             "'prod' and 'len'")
        a_dtype = np.dtype(type(a))
    else:
        a_dtype = a.dtype

    if dtype is not None:
        if np.issubdtype(dtype, np.bool_) and \
                not('all' in func_str or 'any' in func_str):
            raise TypeError("function %s requires a more complex datatype "
                            "than bool" % func_str)
        if not np.issubdtype(dtype, np.integer) and func_str in ('len', 'nanlen'):
            raise TypeError("function %s requires an integer datatype" % func_str)
        return np.dtype(dtype)
    else:
        try:
            return np.dtype(_forced_types[func_str])
        except KeyError:
            if func_str in _forced_float_types:
                if np.issubdtype(a_dtype, np.floating):
                    return a_dtype
                else:
                    return np.dtype(np.float64)
            else:
                if func_str == 'sum':
                    if np.issubdtype(a_dtype, np.int64):
                        return np.dtype(np.int64)
                    elif np.issubdtype(a_dtype, np.integer):
                        maxval = np.iinfo(a_dtype).max * n
                        return minimum_dtype(maxval, a_dtype)
                    elif np.issubdtype(a_dtype, np.bool_):
                        return minimum_dtype(n, a_dtype)
                    else:
                        return a_dtype
                elif func_str in _forced_same_type:
                    return a_dtype
                else:
                    if isinstance(a_dtype, np.integer):
                        return np.dtype(np.int64)
                    else:
                        return a_dtype


def minval(fill_value, dtype):
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.floating):
        return -np.inf
    if issubclass(dtype.type, np.integer):
        return  np.iinfo(dtype).min
    return np.finfo(dtype).min


def maxval(fill_value, dtype):
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.floating):
        return np.inf
    if issubclass(dtype.type, np.integer):
        return  np.iinfo(dtype).max
    return np.finfo(dtype).max


def check_fill_value(fill_value, dtype, func=None):
    if func in ('all', 'any', 'allnan', 'anynan'):
        check_boolean(fill_value)
    else:
        try:
            return dtype.type(fill_value)
        except ValueError:
            raise ValueError("fill_value must be convertible into %s"
                             % dtype.type.__name__)


def check_group_idx(group_idx, a=None, check_min=True):
    if a is not None and group_idx.size != a.size:
        raise ValueError("The size of group_idx must be the same as "
                         "a.size")
    if not issubclass(group_idx.dtype.type, np.integer):
        raise TypeError("group_idx must be of integer type")
    if check_min and np.min(group_idx) < 0:
        raise ValueError("group_idx contains negative indices")


def _ravel_group_idx(group_idx, a, axis, size, order, method="ravel"):
    ndim_a = a.ndim
    size_in = int(np.max(group_idx)) + 1 if size is None else size
    group_idx_in = group_idx
    group_idx = []
    size = []
    for ii, s in enumerate(a.shape):
        if method == "ravel":
            ii_idx = group_idx_in if ii == axis else np.arange(s)
            ii_shape = [1] * ndim_a
            ii_shape[ii] = s
            group_idx.append(ii_idx.reshape(ii_shape))
        size.append(size_in if ii == axis else s)
    if method == "ravel":
        group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                         mode='raise')
    elif method == "offset":
        group_idx = offset_labels(group_idx_in, a.shape, axis, order, size_in)
    return group_idx, size


def offset_labels(group_idx, inshape, axis, order, size):
    """
    Offset group labels by dimension. This is used when we
    reduce over a subset of the dimensions of by. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """

    newaxes = tuple(ax for ax in range(len(inshape)) if ax != axis)
    group_idx = np.broadcast_to(np.expand_dims(group_idx, newaxes), inshape)
    if axis not in (-1, len(inshape) - 1):
        group_idx = np.moveaxis(group_idx, axis, -1)
    newshape = group_idx.shape[:-1] + (-1,)

    group_idx = (group_idx +
                 np.arange(np.prod(newshape[:-1]), dtype=int).reshape(newshape)
                 * size
                 )
    if axis not in (-1, len(inshape) - 1):
        return np.moveaxis(group_idx, -1, axis)
    else:
        return group_idx


def input_validation(group_idx, a, size=None, order='C', axis=None,
                     ravel_group_idx=True, check_bounds=True, method="ravel", func=None):
    """ Do some fairly extensive checking of group_idx and a, trying to
    give the user as much help as possible with what is wrong. Also,
    convert ndim-indexing to 1d indexing.
    """
    if not isinstance(a, (int, float, complex)) and not is_duck_array(a):
        a = np.asanyarray(a)
    if not is_duck_array(group_idx):
        group_idx = np.asanyarray(group_idx)

    if not np.issubdtype(group_idx.dtype, np.integer):
        raise TypeError("group_idx must be of integer type")

    if check_bounds and np.any(group_idx < 0):
        raise ValueError("negative indices not supported")

    ndim_idx = np.ndim(group_idx)
    ndim_a = np.ndim(a)

    if axis is None:
        if ndim_a > 1:
            raise ValueError("a must be scalar or 1 dimensional, use .ravel to"
                             " flatten. Alternatively specify axis.")
    elif axis >= ndim_a or axis < -ndim_a:
        raise ValueError("axis arg too large for np.ndim(a)")
    else:
        axis = axis if axis >= 0 else ndim_a + axis
        if ndim_idx > 1:
            raise NotImplementedError("only 1d indexing currently"
                                      "supported with axis arg.")
        elif a.shape[axis] != len(group_idx):
            raise ValueError("a.shape[axis] doesn't match length of group_idx.")
        elif size is not None and not np.isscalar(size):
            raise NotImplementedError("when using axis arg, size must be"
                                      "None or scalar.")
        else:
            is_form_3 = group_idx.ndim == 1 and a.ndim > 1 and axis is not None
            orig_shape = a.shape if is_form_3 else group_idx.shape
            if isinstance(func, str) and "arg" in func:
                unravel_shape = orig_shape
            else:
                unravel_shape = None

            group_idx, size = _ravel_group_idx(group_idx, a, axis, size, order, method=method)
            flat_size = np.prod(size)
            ndim_idx = ndim_a
            size = orig_shape if is_form_3 and not callable(func) and "cum" in func else size
            return group_idx.ravel(), a.ravel(), flat_size, ndim_idx, size, unravel_shape

    if ndim_idx == 1:
        if size is None:
            size = int(np.max(group_idx)) + 1
        else:
            if not np.isscalar(size):
                raise ValueError("output size must be scalar or None")
            if check_bounds and np.any(group_idx > size - 1):
                raise ValueError("one or more indices are too large for "
                                 "size %d" % size)
        flat_size = size
    else:
        if size is None:
            size = np.max(group_idx, axis=1).astype(int) + 1
        elif np.isscalar(size):
            raise ValueError("output size must be of length %d"
                             % len(group_idx))
        elif len(size) != len(group_idx):
            raise ValueError("%d sizes given, but %d output dimensions "
                             "specified in index" % (len(size),
                                                     len(group_idx)))
        if ravel_group_idx:
            group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                             mode='raise')
        flat_size = np.prod(size)

    if not (np.ndim(a) == 0 or len(a) == group_idx.size):
        raise ValueError("group_idx and a must be of the same length, or a"
                         " can be scalar")

    return group_idx, a, flat_size, ndim_idx, size, None


def unpack(group_idx, ret):
    """ Take an aggregate packed array and uncompress it to the size of group_idx.
        This is equivalent to ret[group_idx].
    """
    return ret[group_idx]


def allnan(x):
    return np.all(np.isnan(x))


def anynan(x):
    return np.any(np.isnan(x))


def nanfirst(x):
    return x[~np.isnan(x)][0]


def nanlast(x):
    return x[~np.isnan(x)][-1]


def multi_arange(n):
    """By example:

        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    That is it is equivalent to something like this :

        hstack((arange(n_i) for n_i in n))

    This version seems quite a bit faster, at least for some
    possible inputs, and at any rate it encapsulates a task
    in a function.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def label_contiguous_1d(X):
    """
    WARNING: API for this function is not liable to change!!!

    By example:

        X =      [F T T F F T F F F T T T]
        result = [0 1 1 0 0 2 0 0 0 3 3 3]

    Or:
        X =      [0 3 3 0 0 5 5 5 1 1 0 2]
        result = [0 1 1 0 0 2 2 2 3 3 0 4]

    The ``0`` or ``False`` elements of ``X`` are labeled as ``0`` in the output. If ``X``
    is a boolean array, each contiguous block of ``True`` is given an integer
    label, if ``X`` is not boolean, then each contiguous block of identical values
    is given an integer label. Integer labels are 1, 2, 3,..... (i.e. start a 1
    and increase by 1 for each block with no skipped numbers.)

    """

    if X.ndim != 1:
        raise ValueError("this is for 1d masks only.")

    is_start = np.empty(len(X), dtype=bool)
    is_start[0] = X[0]

    if X.dtype.kind == 'b':
        is_start[1:] = ~X[:-1] & X[1:]
        M = X
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return L


def relabel_groups_unique(group_idx):
    """
    See also ``relabel_groups_masked``.

    keep_group:  [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
    ret:         [0 3 3 3 0 2 4 2 0 1 1 0 3 4 4]

    Description of above: unique groups in input was ``1,2,3,5``, i.e.
    ``4`` was missing, so group 5 was relabled to be ``4``.
    Relabeling maintains order, just "compressing" the higher numbers
    to fill gaps.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(group_idx, keep_group):
    """
    group_idx: [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]

                 0 1 2 3 4 5
    keep_group: [0 1 0 1 1 1]

    ret:       [0 2 2 2 0 0 4 0 0 1 1 0 2 4 4]

    Description of above in words: remove group 2, and relabel group 3,4, and 5
    to be 2, 3 and 4 respecitvely, in order to fill the gap.  Note that group 4 was never used
    in the input group_idx, but the user supplied mask said to keep group 4, so group
    5 is only moved up by one place to fill the gap created by removing group 2.

    That is, the mask describes which groups to remove,
    the remaining groups are relabled to remove the gaps created by the falsy
    elements in ``keep_group``.  Note that ``keep_group[0]`` has no particular meaning because it refers
    to the zero group which cannot be "removed".

    ``keep_group`` should be bool and ``group_idx`` int.
    Values in ``group_idx`` can be any order, and
    """

    keep_group = keep_group.astype(bool, copy=not keep_group[0])
    if not keep_group[0]: 
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return relabel[group_idx]

def is_duck_array(value):
    """
    This function was copied from xarray/core/utils.py under the terms
    of Xarray's Apache-2 license
    """
    if isinstance(value, np.ndarray):
        return True
    return (
            hasattr(value, "ndim")
            and hasattr(value, "shape")
            and hasattr(value, "dtype")
            and hasattr(value, "__array_function__")
            and hasattr(value, "__array_ufunc__")
    )


def iscomplexobj(x):
    """
    Copied from np.iscomplexobj so that we place fewer requirements
    on duck array types.
    """
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = np.asarray(x).dtype.type
    return issubclass(type_, np.complexfloating)




'''
************** numba aggregate***********
'''

class AggregateOp(object):
    """
    Every subclass of AggregateOp handles a different aggregation operation. There are
    several private class methods that need to be overwritten by the subclasses
    in order to implement different functionality.
    On object instantiation, all necessary static methods are compiled together into
    two jitted callables, one for scalar arguments, and one for arrays. Calling the
    instantiated object picks the right cached callable, does some further preprocessing
    and then executes the actual aggregation operation.
    """

    forced_fill_value = None
    counter_fill_value = 1
    counter_dtype = bool
    mean_fill_value = None
    mean_dtype = np.float64
    outer = False
    reverse = False
    nans = False

    def __init__(self, func=None, **kwargs):
        if func is None:
            func = type(self).__name__.lower()
        self.func = func
        self.__dict__.update(kwargs)
        # Cache the compiled functions, so they don't have to be recompiled on every call
        self._jit_scalar = self.callable(self.nans, self.reverse, scalar=True)
        self._jit_non_scalar = self.callable(self.nans, self.reverse, scalar=False)

    def __call__(self, group_idx, a, size=None, fill_value=0, order='C',
                 dtype=None, axis=None, ddof=0):
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func=self.func)
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        check_fill_value(fill_value, dtype, func=self.func)
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, counter, mean, outer = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        if not np.isscalar(a):
            a = np.ascontiguousarray(a)
            jitfunc = self._jit_non_scalar
        else:
            jitfunc = self._jit_scalar
        jitfunc(group_idx, a, ret, counter, mean, outer, fill_value, ddof)
        self._finalize(ret, counter, fill_value)

        if self.outer:
            ret = outer

        if ndim_idx > 1:
            if unravel_shape is not None:
                mask = ret == fill_value
                ret[mask] = 0
                ret = np.unravel_index(ret, unravel_shape)[axis]
                ret[mask] = fill_value
            ret = ret.reshape(size, order=order)
        return ret

    @classmethod
    def _initialize(cls, flat_size, fill_value, dtype, input_dtype, input_size):
        if cls.forced_fill_value is None:
            ret = np.full(flat_size, fill_value, dtype=dtype)
        else:
            ret = np.full(flat_size, cls.forced_fill_value, dtype=dtype)

        counter = mean = outer = None
        if cls.counter_fill_value is not None:
            counter = np.full_like(ret, cls.counter_fill_value, dtype=cls.counter_dtype)
        if cls.mean_fill_value is not None:
            dtype = cls.mean_dtype if cls.mean_dtype else input_dtype
            mean = np.full_like(ret, cls.mean_fill_value, dtype=dtype)
        if cls.outer:
            outer = np.full(input_size, fill_value, dtype=dtype)

        return ret, counter, mean, outer

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        if cls.forced_fill_value is not None and fill_value != cls.forced_fill_value:
            if cls.counter_dtype == bool:
                ret[counter] = fill_value
            else:
                ret[~counter.astype(bool)] = fill_value

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        """ Compile a jitted function doing the hard part of the job """
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = nb.njit(_valgetter)
        outersetter = nb.njit(cls._outersetter)

        if not nans:
            inner = nb.njit(cls._inner)
        else:
            cls_inner = nb.njit(cls._inner)
            cls_nan_check = nb.njit(cls._nan_check)

            @nb.njit
            def inner(ri, val, ret, counter, mean, fill_value):
                if not cls_nan_check(val):
                    cls_inner(ri, val, ret, counter, mean, fill_value)

        @nb.njit
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # ddof needs to be present for being exchangeable with loop_2pass
            size = len(ret)
            rng = range(len(group_idx) - 1, -1, -1) if reverse else range(len(group_idx))
            for i in rng:
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                inner(ri, val, ret, counter, mean, fill_value)
                outersetter(outer, i, ret[ri])

        return loop

    @staticmethod
    def _valgetter(a, i):
        return a[i]

    @staticmethod
    def _valgetter_scalar(a, i):
        return a

    @staticmethod
    def _nan_check(val):
        return val != val

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        raise NotImplementedError("subclasses need to overwrite _inner")

    @staticmethod
    def _outersetter(outer, i, val):
        pass


class Aggregate2pass(AggregateOp):
    """Base class for everything that needs to process the data twice like mean, var and std."""
    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        loop_1st = super(Aggregate2pass, cls).callable(nans=nans, reverse=reverse, scalar=scalar)

        _2pass_inner = nb.njit(cls._2pass_inner)

        @nb.njit
        def loop_2nd(ret, counter, mean, fill_value, ddof):
            for ri in range(len(ret)):
                if counter[ri] > ddof:
                    ret[ri] = _2pass_inner(ri, ret, counter, mean, ddof)
                else:
                    ret[ri] = fill_value

        @nb.njit
        def loop_2pass(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            loop_1st(group_idx, a, ret, counter, mean, outer, fill_value, ddof)
            loop_2nd(ret, counter, mean, fill_value, ddof)

        return loop_2pass

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        raise NotImplementedError("subclasses need to overwrite _2pass_inner")

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        """Copying the fill value is already done in the 2nd pass"""
        pass


class AggregateNtoN(AggregateOp):
    """Base class for cumulative functions, where the output size matches the input size."""
    outer = True

    @staticmethod
    def _outersetter(outer, i, val):
        outer[i] = val


class AggregateGeneric(AggregateOp):
    """Base class for jitting arbitrary functions."""
    counter_fill_value = None

    def __init__(self, func, **kwargs):
        self.func = func
        self.__dict__.update(kwargs)
        self._jitfunc = self.callable(self.nans)

    def __call__(self, group_idx, a, size=None, fill_value=0, order='C',
                 dtype=None, axis=None, ddof=0):
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        check_fill_value(fill_value, dtype, func=self.func)
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, _, _, _ = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        sortidx = np.argsort(group_idx, kind='mergesort')
        self._jitfunc(sortidx, group_idx, a, ret)

        if ndim_idx > 1:
            ret = ret.reshape(size, order=order)
        return ret

    def callable(self, nans=False):
        """Compile a jitted function and loop it over the sorted data."""
        func = nb.njit(self.func)

        @nb.njit
        def loop(sortidx, group_idx, a, ret):
            size = len(ret)
            group_idx_srt = group_idx[sortidx]
            a_srt = a[sortidx]

            indices = step_indices(group_idx_srt)
            for i in range(len(indices) - 1):
                start_idx, stop_idx = indices[i], indices[i + 1]
                ri = group_idx_srt[start_idx]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                ret[ri] = func(a_srt[start_idx:stop_idx])

        return loop


class Sum(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += val


class Prod(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] *= val


class Len(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += 1


class All(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= bool(val)


class Any(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= bool(val)


class Last(AggregateOp):
    counter_fill_value = None

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        ret[ri] = val


class First(Last):
    reverse = True


class AllNan(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= val != val


class AnyNan(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= val != val


class Max(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        if counter[ri]:
            ret[ri] = val
            counter[ri] = 0
        elif ret[ri] < val:
            ret[ri] = val


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        if counter[ri]:
            ret[ri] = val
            counter[ri] = 0
        elif ret[ri] > val:
            ret[ri] = val


class ArgMax(AggregateOp):
    mean_fill_value = np.nan

    @staticmethod
    def _valgetter(a, i):
        return a[i], i

    @staticmethod
    def _nan_check(val):
        return val[0] != val[0]

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        cmp_val, arg = val
        if counter[ri]:
            counter[ri] = 0
            mean[ri] = cmp_val
            if cmp_val == cmp_val:
                ret[ri] = arg
        elif mean[ri] < cmp_val:
            mean[ri] = cmp_val
            ret[ri] = arg
        elif cmp_val != cmp_val:
            mean[ri] = cmp_val
            ret[ri] = fill_value


class ArgMin(ArgMax):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        cmp_val, arg = val
        if counter[ri]:
            counter[ri] = 0
            mean[ri] = cmp_val
            if cmp_val == cmp_val:
                ret[ri] = arg
        elif mean[ri] > cmp_val:
            mean[ri] = cmp_val
            ret[ri] = arg
        elif cmp_val != cmp_val:
            mean[ri] = cmp_val
            ret[ri] = fill_value


class SumOfSquares(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += val * val


class Mean(Aggregate2pass):
    forced_fill_value = 0
    counter_fill_value = 0
    counter_dtype = int

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] += 1
        ret[ri] += val

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        return ret[ri] / counter[ri]


class Std(Mean):
    mean_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] += 1
        mean[ri] += val
        ret[ri] += val * val

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        mean2 = mean[ri] * mean[ri]
        return np.sqrt((ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof))


class Var(Std):
    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        mean2 = mean[ri] * mean[ri]
        return (ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof)


class CumSum(AggregateNtoN, Sum):
    pass


class CumProd(AggregateNtoN, Prod):
    pass


class CumMax(AggregateNtoN, Max):
    pass


class CumMin(AggregateNtoN, Min):
    pass


def get_funcs():
    funcs = dict()
    for op in (Sum, Prod, Len, All, Any, Last, First, AllNan, AnyNan, Min, Max,
               ArgMin, ArgMax, Mean, Std, Var, SumOfSquares,
               CumSum, CumProd, CumMax, CumMin):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in funcs_no_separate_nan:
            funcname = 'nan' + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


_impl_dict = get_funcs()
_default_cache = {}


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, cache=True, **kwargs):
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        if cache in (None, False):
            aggregate_op = AggregateGeneric(func)
        else:
            if cache is True:
                cache = _default_cache
            aggregate_op = cache.setdefault(func, AggregateGeneric(func))
        return aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)


aggregate.__doc__ = """
    This is the numba implementation of aggregate.
    """


@nb.njit
def step_count(group_idx):
    """Return the amount of index changes within group_idx."""
    cmp_pos = 0
    steps = 1
    if len(group_idx) < 1:
        return 0
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            steps += 1
    return steps


@nb.njit
def step_indices(group_idx):
    """Return the edges of areas within group_idx, which are filled with the same value."""
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, np.int64)
    indices[0] = 0
    indices[-1] = group_idx.size
    cmp_pos = 0
    ri = 1
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[ri] = i
            ri += 1
    return indices



