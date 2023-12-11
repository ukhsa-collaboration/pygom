==============================================
pygom - ODE modelling in Python -Documentation
==============================================

Procedure For Building HTML Documentation
-----------------------------------------

Any new documentation should be in the from of an ipynb jupyter
notebook. This should be placed in the /notebooks sub directoy 
of the /docs folder.

The name of the new file sluod be added to _toc.yml in the /docs
subdirectory.

The following command builds the html:

jupyter-book build --all -v docs/


The html is created in the /docs subfolder  /_build
