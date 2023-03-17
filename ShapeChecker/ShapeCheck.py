import einops, re

class ShapeCheck :
    def __init__(self, shapes=None, string=None) :
        self.shape_dict = {}
        if shapes and string :
            self.update(shapes, string)

    @staticmethod
    def clean(string) :
        string = re.sub(' +', ' ', string)
        for i in ['1', '(', ')', ' ->'] :
            string = string.replace(i,'')
        return string

    def update(self, shapes, string) :
        """
        Update internal dictionary with provided dimensions
            shapes list : list of shape to build a dict separated by spaces ex : [1,2,3]
            string : string with the name of the dimensions ex : 'a b c'
        """
        string = self.clean(string).split(' ')
        assert len(string) == len(shapes), 'Error in number of dimensions shapes'
        for h,s in zip(shapes, string) :
            if s in self.shape_dict :
                assert self.shape_dict[s] == h,\
                f'Error in ShapeCheck : {s}={self.shape_dict[s]} got {h}'
            else :
                self.shape_dict[s] =  h

    def _get(self, string) :
        """
        Return a dict with all shape in string
        """
        string = self.clean(string).split(' ')
        return {s : self.shape_dict[s] for s in string if s in self.shape_dict}

    def get(self, string) :
        """
        Return a dict with all shape in string
        Raising error if a request is not known
        """
        string = self.clean(string).split(' ')
        return {s : self.shape_dict[s] for s in string}


    def rearrange(self, tensor, string, **kwargs) :
        """
        Apply einops repeat with the sc control
        """
        return einops.rearrange(tensor, string, **self._get(string), **kwargs)

    def repeat(self, tensor, string, **kwargs) :
        """
        Apply einops repeat with the sc control
        """
        return einops.repeat(tensor, string, **self._get(string), **kwargs)

    def reduce(self, tensor, string, reduction, **kwargs) :
        """
        Apply einops repeat with the sc control
        """
        return einops.reduce(tensor, string, reduction, **self._get(string), **kwargs)
