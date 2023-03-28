import sys
sys.path.append('../')
from my_layers import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # input
    to_input( '4genos.png', width=3, height=24), # might need jpg...

    # downsampling/encoding layers
    to_Conv("conv0", 64, 500000, offset="(0,0,0)", to="(0,0,0)", height=120, depth=10, width=1 ),
    to_Pool("pool0", offset="(1,0,0)", to="(conv0-east)", height=60, depth=10, width=1),

    to_Conv("conv1", 108, 49999, offset="(3,0,0)", to="(pool0-east)", height=60, depth=20, width=1 ),
    to_connection( "pool0", "conv1"),
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=30, depth=20, width=1),

    to_Dense("dense0", offset="(3,0,0)", to="(pool1-east)", height=5, depth=30, width=1),
    to_connection("pool1", "dense0"),

    
    # flatten  + dense
    to_Dense("flatten", 1, 9604, offset="(2,0,0)", to="(dense0-east)", height=40, depth=1, width=1 ),
    to_connection( "dense0", "flatten"),

    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

