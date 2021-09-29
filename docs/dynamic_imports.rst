Dynamic imports
===============

For its users, ``serotiny`` intends to be a configuration-centric framework.
The idea is that for someone to leverage a ``serotiny``-based workflow and
adapt it to their use-case, they should only have to tweak configuration
parameters in a YAML file. Moreover, this makes it easy to integrate with
workflow frameworks like `Snakemake <https://snakemake.readthedocs.io/en/stable/>`_ ,
which is also YAML/config-based

To maximize the configurability of ``serotiny``, we developed a dynamic import
functionality, which allows Python dictionaries (which can be read from
YAML configs) containing special configurations to be interpreted and loaded.

There are 3 ways of using this functionality:

* ``^init`` - to instantiate classes

* ``^invoke`` - to return the results of function calls

* ``^bind`` - to load functions with partially attributed arguments

Additionally, there is a special keyword ``^positional_args`` which
can be used when the arguments to be served must be positional, or
the function takes a variable number of arguments.

Examples for each of these are provided ahead.

``^init``
*************

To instantiate a class described by a dictionary, we can use the ``^init``
keyword. For example, to instantiate a Unet:
::

   >>> net_config = yaml.full_load("""
   ^init: serotiny.networks._3d.Unet
   depth: 4
   channel_fan: 2
   channel_fan_top: 8 # down from 64
   num_input_channels: 2 # specify how many channels
   num_output_channels: 1
   """)
   >>> net = load_config(net_config)


``^invoke``
***************

To call and return the result of a function, the ``^invoke`` keyword is used:
::

   >>> ones_config = yaml.full_load("""
   ^invoke: numpy.ones
   shape: [3,3]
   """)
   >>> ones = load_config(ones_config)
   >>> ones
   array([[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]])



``^bind``
*************

Let's use ``numpy.random.randint`` as an example.
This is its signature
::

   >>> numpy.random.randint(low, high=None, size=None, dtype='l')

We can bind the ``size`` and ``low`` parameters, yielding a function that expects only a ``high``
parameter
::

   >>> randint_config = yaml.full_load("""
   ^bind: numpy.random.randint
   low: 0
   size: 5
   """)
   >>> my_randint = load_config(randint_config)

If we now call the function, we obtain:
::

   >>> my_randint(high=20)
   array([ 4, 19,  7,  1,  7])

Note: we can use ``^bind`` with no arguments to simply pass the original function.
We can also bind all of the arguments of a function, thus yielding a function
which needs no further arguments to be called:
::

   >>> randn_config = yaml.full_load("""
   ^bind: numpy.random.randint
   low: 0
   high: 5
   size: 5
   """)
   >>> my_randint = load_config(randint_config)
   >>> my_randint()
   array([1, 4, 3, 1, 3])


``^positional_args``
************************

When the function of interest has a variable number of arguments, we can use
the special keyword ``^positional_args`` to specify a list of positional arguments:
::

   >>> randn_config = yaml.full_load("""
   ^bind: numpy.random.randn
   ^positional_args: [10, 3]
   """)
   >>> my_randn = load_config(randn_config)
   >>> my_randn()
   array([[-0.8956135 ,  0.32854592,  0.20821983],
          [-1.59323792,  0.0151588 ,  0.05956044],
          [ 1.56047808,  1.3287971 , -1.78241693],
          [-0.98165106, -0.59621   ,  0.98945791],
          [-0.37383497,  0.49404687,  2.14726909],
          [ 0.98437869, -0.66560783,  0.00638175],
          [-1.89774285, -0.60767339, -0.78599943],
          [ 0.32604151,  0.48772363, -0.68928192],
          [-0.98970493, -0.29300648,  0.01525316],
          [-0.54736776, -0.41411613,  0.82680974]])
