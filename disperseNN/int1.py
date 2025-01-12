import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    #input                                                                                                                                                     
    to_Conv("d1", 1, 384, offset="(0,0,0)", to="(0,0,0)", height=30, depth=1, width=1 ),
    to_Conv("d2", 1, 128, offset="(1,0,0)", to="(d1-east)", height=10, depth=1, width=1 ),
    to_connection( "d1", "d2"),


    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

