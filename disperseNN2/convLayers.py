import sys
sys.path.append('../')
from my_layers import *
import numpy as np



def my_trans(x):
    return x**(1/2) 

scal = 1 # multiply this by the transformed layer sizes

ds = [[None,None],
      [5000,2],
      [64,4999],
      [64,499],
      [108,498],
      [108,49],
      [108,49],
      [1,5292],
      ]

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # input
    to_input( 'genos_flip.png', width=1.5, height=14), # might need jpg...

    # invisible layer, only for drawing a connection to the input image
    to_Conv("input", "", "", offset="(-2.5,0,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

    # another invisible layer, for getting layer-size labels onto the input image
    to_Conv("input_lab", "", ds[1][0], offset="(-3.7,-7.5,0)", to="(0,0,0)", height=0, depth=0, width=0 ),
    to_Conv("input_lab", "", ds[1][1], offset="(-2.8,-7.1,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

    # downsampling/encoding layers
    to_Conv("conv0", ds[2][0], ds[2][1], offset="(-1,0,0)", to="(0,0,0)", height=my_trans(ds[2][1])*scal, depth=my_trans(ds[2][0])*scal, width=1 ),
    to_connection( "input", "conv0"),
    to_Pool("pool0", ds[3][0], ds[3][1], offset="(0.5,0,0)", to="(conv0-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1), 

    to_Conv("conv1", ds[4][0], ds[4][1], offset="(1,0,0)", to="(pool0-east)",height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ),
    to_connection( "pool0", "conv1"),
    to_Pool("pool1", ds[5][0], ds[5][1], offset="(0.5,0,0)", to="(conv1-east)", height=my_trans(ds[5][1])*scal, depth=my_trans(ds[5][0])*scal, width=1),

    to_Dense("dense0", ds[6][0], ds[6][1], offset="(1,0,0)", to="(pool1-east)", height=my_trans(ds[6][1])*scal, depth=my_trans(ds[6][0])*scal, width=1), 
    to_connection("pool1", "dense0"),

    
    # flatten  + densep
    to_Dense("flatten", ds[7][0], ds[7][1], offset="(1,0,0)", to="(dense0-east)", height=my_trans(ds[7][1])*scal, depth=my_trans(ds[7][0])*scal, width=1 ),
    to_connection( "dense0", "flatten"),


    # # locs
    # to_Dense("locs", 2, 2, offset="(2,0,0)", to="(flatten-south)", height=5, depth=5, width=1),
    
    
    
    # # feature block
    # to_Dense("feature_block", 45, 5292, offset="(5,0,0)", to="(flatten-east)", height=120, depth=45, width=1 ),








    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

