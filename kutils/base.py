"""
Basic Imports for all scripts and notebooks
"""
from collections import defaultdict
from termcolor import colored
import pickle

from IPython.display import display, HTML, Markdown

# Termcolor Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')

class Dict(defaultdict): 
    def __init__(self, base_dict={}): 
        super().__init__()
        self._add_base_dict_to_self(base_dict)
        
    def _add_base_dict_to_self(self, base_dict): 
        for key, val in base_dict.items(): 
            self[key] = val

    def __getattr__(self, key):
        if key in self: 
            return self[key]
        else: 
            raise Exception(f'{key} not found in the object')

    def __setattr__(self, key, value):
        self[key] = value
    
    def __repr__(self):
        dict_key_color = 'red'
        dict_val_color = 'blue'
        res = '\n'
        for key, value in self.items(): 
            k, v = str(key)[:512], str(value)[:512]
            if isinstance(value, Dict): 
                space = min(8, len(k)+1)
                v = v.replace('\n', '\n' + ' '*space)
                k = '\n' + colored(k, dict_key_color, attrs=['bold'])
            else: 
                k = colored(k, dict_key_color)
            line = k + ': ' + colored(v, dict_val_color)
            res += line + '\n'
        return res

    def save(self, path): 
        with open(path, 'wb') as f:
            pickle.dump(self, f)        

    @staticmethod
    def load(path): 
        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file

    def _heading(self, title, level=3): 
        margin = 'margin-left:5px;'
        try: 
            text = title.title()
            html = f"<h{level} style='text-align:center; {margin}'> {text} </h{level}> <hr/>" 
            print()
            display(Markdown(html))
        except Exception as e: 
            print(e)

    def display(self, title): 
        self._heading(title)
        print(self)