import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    #input                                                                                                                                                     
    to_input( 'pos.png', to='(-1,0,0)', width=1, height=16), # might need jpg...                                                                                           
    to_Conv("d1", 1, 256, offset="(0,0,0)", to="(0,0,0)", height=25, depth=1, width=1 ),

    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

