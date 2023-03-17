ShapeChecker
==========

This small package (~60 lines of code comments included) is an extension
to awesome `einops <https://einops.rocks/>`__ and your new tool to
assist you when doing tensors manipulation.

ShapeCheck can help you to :

1. Control that you have no dimension-related errors in your code
2. Call ``rearrange``, ``reduce`` or ``repeat`` without having to
   specify the dimensions
3. Share your dictionary of dimensions through your code.

Basic Usage
-----------

.. code:: python

   from ShapeChecker import ShapeCheck

   def myfunction(x) :
       sc = ShapeCheck() # Instantiate your ShapeCheck object
       sc.update(x.shape, 'batch_size c h w') # sc can hold your dimension in memory
       sc.update([3], 'colors') # You can use any list to hold dimension shapes

       x_flat = sc.rearrange(x, 'batch_size c h w -> batch_size (c h w)')
       x_original = sc.rearrange(x_flat, 'batch_size (c h w) -> batch_size c h w') # No for any specifications for dimensions !
       assert (torch.equal(x, x_original))

       x_repeat = sc.repeat(x, 'batch_size c h w -> k batch_size c h w', k=5) # Flexibility
       return sc

   def myotherfunction(sc, y) : # you can pass sc as an argument and use it in your code
       sc.update(y.shape, 'batch_size colors features h w') # automatically check that the shapes corresponds !
       y_reduce = sc.reduce(y, 'batch_size colors features h w -> batch_size (colors features)', 'mean')

       print("The batch size is {batch_size} and features dim is {features}".format(**sc.get('batch_size features'))) # Access saved information any time


   x = torch.rand(10, 5, 40, 50)
   y = torch.rand(10, 3, 25, 40, 50)
   sc = myfunction(x)
   myotherfunction(sc, y)

Documentation
-------------

-  ``ShapeCheck(list, str) -> sc object`` : initialize ``sc`` object and
   call ``sc.update(list, str)``
-  ``sc.update(list, str)`` : Update internal dictionary with provided
   dimensions

   -  ``shapes`` list : list of shape to build a dict separated by
      spaces ex : [1,2,3]
   -  ``string`` : string with the name of the dimensions ex : 'a b c'

-  ``sc.[rearrange/reduce/repeat](tensor, pattern)`` : call
   ``einops.[rearrange/reduce/repeat]`` indicating necessary dimension
   as ``axes_lengths``
-  ``sc.get(str)`` : return a dictionary ``{key:shape}`` for all keys in
   str
