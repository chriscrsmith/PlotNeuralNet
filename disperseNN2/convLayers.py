import sys
sys.path.append('../')
from my_layers import *
import numpy as np



def my_trans(x):
    return x**(1/2) 

scal = 1 # multiply this by the transformed layer sizes

ds = [[None,None], # 0
      [5000,2], # 1
      [64,4999], # 2
      [64,499], # 3
      [108,498], # 4
      [108,49], # 5
      [1,5292], # 6
      [1,128], # 7
      [None, None], # 8
      [1,1], # 9
      [45, 128], # 10
      [45, 1], # 11
      [45, 128], # 12                                                                                                                                                          
      [45, 128], # 13
      ]

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # input
    to_input( 'genos_flip.png', width=1, height=14), # might need jpg...

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

    # feature vector
    to_input( 'locs_flip.png', to='(1,5.25,0)', width=4, height=1),
    to_DensePurp("locs", "","", offset="(-1,4.75)", to="(pool1-north)", height=0, depth=0, width=0), # invis
    to_DensePurp("locs2", ds[9][0], ds[9][1], offset="(0,6.675,0)", to="(pool1-north)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),
    to_connection( "locs", "locs2"),
    to_Dense("pencil", ds[9][0], ds[6][1]+ds[9][1], offset="(1,0,0)", to="(pool1-east)", height=my_trans(ds[6][1])*scal, depth=my_trans(ds[6][0])*scal, width=1),
    to_connection( "pool1", "pencil"),
    #to_DensePurp("eraser", ds[9][0], ds[6][1]+ds[9][1], offset="(-0.1,-0.1,0)", to="(pencil-south)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),
    to_DensePurp("eraser", "","", offset="(-0.1,0.1,0)", to="(pencil-north)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),   
    to_connection( "locs2", "eraser"),
    to_Dense("dense0", ds[7][0], ds[7][1], offset="(0.75,0,0)", to="(pencil-east)", height=my_trans(ds[7][1])*scal, depth=my_trans(ds[7][0])*scal, width=1),
    to_connection("pencil", "dense0"),                                                                                                                       
    
    # invisible layer for some connections
    to_Dense("invis1", "","", offset="(1,0,0)", to="(dense0-east)", height=0, depth=0, width=0),
    to_connection( "dense0", "invis1"),    
    to_Dense("invis2", "","", offset="(3,0,0)", to="(invis1-east)", height=0, depth=0, width=0),
            
    # feature block
    to_Dense("feature_block", ds[10][0], ds[10][1], offset="(1,0,0)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_connection( "invis2", "feature_block"),
    #to_DensePurp("locs3", ds[11][0], ds[10][1]+ds[11][1], offset="(-0.1,-0.1,0)", to="(feature_block-south)", height=my_trans(ds[11][1])*scal, depth=my_trans(ds[11][0])*scal, width=1),

    # dense and output
    to_Dense("dense1", ds[12][0], ds[12][1], offset="(0.75,0,0)", to="(feature_block-east)", height=my_trans(ds[12][1])*scal, depth=my_trans(ds[12][0])*scal, width=1),
    to_connection( "feature_block", "dense1"),
    # to_Dense("dense2", ds[13][0], ds[13][1], offset="(0.75,0,0)", to="(dense1-east)", height=my_trans(ds[13][1])*scal, depth=my_trans(ds[13][0])*scal, width=1),
    # to_connection( "dense1", "dense2"),
    to_Dense("output", 1, 1, offset="(0.75,0,0)", to="(dense1-east)", height=my_trans(1)*scal, depth=my_trans(1)*scal, width=1),
    to_connection( "dense1", "output"),





    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

