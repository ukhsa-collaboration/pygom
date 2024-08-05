# Converting equations into transitions

As seen previously in {doc}`transition`, we can define a model via transitions or explicitly as ODEs.
There may be times when importing a model from elsewhere and the only available information are the ODEs themselves.
If it is known that the ODEs come from some underlying transitions, we provide the functionality to do this automatically.
Of course there is some interpretation...

Here we demostrate usage of this feature via examples of increasing complexity:
{doc}`../notebooks/unroll/unrollSimple`
{doc}`../notebooks/unroll/unrollBD`
{doc}`../notebooks/unroll/unrollHard`
