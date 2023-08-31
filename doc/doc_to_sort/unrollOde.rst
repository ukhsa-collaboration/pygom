.. _unrollOde:

****************************
Convert ODE into transitions
****************************

As seen previously in :ref:`transition`, we can define the model via the transitions or explicitly as ODEs.  There are times when we all just want to test out some model in a paper and the only available information are the ODEs themselves.  Even though we know that the ODEs come from some underlying transitions, breaking them down can be a time consuming process.  We provide the functionalities to do this automatically.  

.. toctree::

    unroll/unrollSimple.rst
    unroll/unrollBD.rst
    unroll/unrollHard.rst
