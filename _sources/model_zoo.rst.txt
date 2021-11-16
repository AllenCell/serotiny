Model zoo
=========

If you use ``serotiny`` to train your models, likely you are
using a checkpoint callback to store your models. By default,
this will store models in a folder inside your home directory
(if you have a ``.cache`` folder): ``~/.cache/serotiny/zoo``.

You can also specify a zoo root by setting the environment variable
``SEROTINY_ZOO_ROOT`` to the desired value. Otherwise, it's also
possible to specify a zoo root by setting the ``path`` parameter
of the ``model_zoo`` config passed to the model training CLI util.

Once a zoo root is specified, the model training functionality will
store models in that folder, under a subfolder corresponding to their
model class. Each model is identified within that subfolder by its
version string.

Loading a trained model
***********************

After a model is trained, we can load it via
:py:func:`serotiny.models.zoo.get_model`, which takes
as parameters: the model class, the version string, and optionally
the zoo root (which can be specified as described in the previous section).

An example would be:
::

    >>> model = get_model(
    ...     model_class="serotiny.models.vae.TabularVAE",
    ...     version_string="MY_VERSION_STRING",
    ...     # optionally we can set the path to the zoo root here,
    ...     # otherwise it is determined as described above
    ...     # zoo_root="/path/to/my/zoo/root"
    ... )
