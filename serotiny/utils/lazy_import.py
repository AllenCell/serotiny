import sys
import operator
import importlib.util
import inspect
from collections import defaultdict

class _lazy_import_from:
    """
    adapted from https://coderbook.com/python/2020/04/23/how-to-make-lazy-python.html
    """
    def __init__(self, module, name):
        # Assign using __dict__ to avoid the setattr method.

        self.__dict__['_wrapped'] = None
        self.__dict__['_is_init'] = False
        self.__dict__['_module'] = module
        self.__dict__['_name'] = name

    def _setup(self):
        self._wrapped = getattr(self._module, self._name)
        self._is_init = True

    def new_method_proxy(func):
        """
        Util function to help us route functions
        to the nested object.
        """
        def inner(self, *args , **kwargs):
            if not self._is_init:
                self._setup()
            return func(self._wrapped, *args, **kwargs)
        return inner

    def __call__(self, *args, **kwargs):
        if not self._is_init:
            self._setup()

        return self._wrapped(*args, **kwargs)

    def __setattr__(self, name, value):
        # These are special names that are on the LazyObject.
        # every other attribute should be on the wrapped object.
        if name in {"_is_init", "_wrapped"}:
            self.__dict__[name] = value
        else:
            if not self._is_init:
                self._setup()
            setattr(self._wrapped, name, value)

    def __delattr__(self, name):
        if name == "_wrapped":
            raise TypeError("can't delete _wrapped.")
        if not self._is_init:
                self._setup()
        delattr(self._wrapped, name)

    __getattr__ = new_method_proxy(getattr)
    __bytes__ = new_method_proxy(bytes)
    __str__ = new_method_proxy(str)
    __bool__ = new_method_proxy(bool)
    __dir__ = new_method_proxy(dir)
    __hash__ = new_method_proxy(hash)
    __class__ = property(new_method_proxy(operator.attrgetter("__class__")))
    __name__ = property(new_method_proxy(operator.attrgetter("__name__")))
    __eq__ = new_method_proxy(operator.eq)
    __lt__ = new_method_proxy(operator.lt)
    __gt__ = new_method_proxy(operator.gt)
    __ne__ = new_method_proxy(operator.ne)
    __hash__ = new_method_proxy(hash)
    __getitem__ = new_method_proxy(operator.getitem)
    __setitem__ = new_method_proxy(operator.setitem)
    __delitem__ = new_method_proxy(operator.delitem)
    __iter__ = new_method_proxy(iter)
    __len__ = new_method_proxy(len)
    __contains__ = new_method_proxy(operator.contains)


class lazy_import:
    def __init__(self, name):
        spec = importlib.util.find_spec(name)
        loader = importlib.util.LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        self.module = module

    def __getattr__(self, attr):
        return _lazy_import_from(self.module, attr)
