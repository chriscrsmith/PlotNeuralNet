import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    to_Conv("conv1", 64, 500000, offset="(0,0,0)", to="(0,0,0)", height=80, depth=10, width=1 ),
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=80, depth=10, width=1),
    to_Conv("conv2", 108, 49999, offset="(3,0,0)", to="(pool1-east)", height=60, depth=20, width=1 ),
    to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)", height=60, depth=20, width=1),
    to_Conv("conv3", 152, 4999, offset="(3,0,0)", to="(pool2-east)", height=40, depth=30, width=1 ),
    to_Pool("pool3", offset="(1,0,0)", to="(conv3-east)", height=40, depth=30, width=1 ),
    to_Conv("conv4", 196, 499, offset="(3,0,0)", to="(pool3-east)", height=20, depth=40, width=1 ),
    to_Pool("pool4", offset="(1,0,0)", to="(conv4-east)", height=20, depth=40, width=1),

    

    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

