import sys
sys.path.append('../')
from my_layers import *
import numpy as np



def my_trans(x):
    return x**(1/2) 

scal = 1 # multiply this by the transformed layer sizes

ds = [[45, 5292], # feature_block1
      [45,4], # locs1
      [10,10], # dense0
      [100,100], # upsample1
      #[100, 5296], # feature_block2
      [(np.sqrt(100)-np.sqrt(45))**2, (np.sqrt(5292)+np.sqrt(4))**2], # padding2
      ]

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # feature block
    to_Dense("feature_block1", "", "", offset="(1,0,0)", height=my_trans(ds[0][1])*scal, depth=my_trans(ds[0][0])*scal, width=1 ),
    to_DensePurp("locs1", ds[1][0], ds[0][1]+ds[1][1], offset="(-0.1,-0.2,0)", to="(feature_block1-south)", height=my_trans(ds[1][1])*scal, depth=my_trans(ds[1][0])*scal, width=1),

    # map 1
    to_Dense("dense0", ds[2][0], ds[2][1], offset="(1,0,0)", to="(feature_block1-east)", height=my_trans(ds[2][1])*scal, depth=my_trans(ds[2][0])*scal, width=1),
    to_connection("feature_block1", "dense0"),                                                                                                                       

    # upsample
    to_Dense("upsample1", ds[3][0], ds[3][1], offset="(1,0,0)", to="(dense0-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),
    to_connection("dense0", "upsample1"),

    # feature block
    to_Dense("feature_block2", "", "", offset="(1,0,0)", to="(upsample1-east)", height=my_trans(ds[0][1])*scal, depth=my_trans(ds[0][0])*scal, width=1 ),                            
    to_connection("upsample1", "feature_block2"),                                                                                                                                    
    to_DensePurp("locs2", ds[1][0], ds[0][1]+ds[1][1], offset="(-0.1,-0.2,0)", to="(feature_block2-south)", height=my_trans(ds[1][1])*scal, depth=my_trans(ds[1][0])*scal, width=1),
    to_Dense("padding2", "", "", offset="(-0.475, -0.475, -1.75)", to="(feature_block2-east)", height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ), # SUPER painstaking
    to_Dense("upsample2", ds[3][0], ds[3][1], offset="(-0.475, -8.75, 0)", to="(padding2-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),
   
   
    
    
    
    
#     # input
#     to_input( 'genos_flip.png', width=1, height=14), # might need jpg...

#     # invisible layer, only for drawing a connection to the input image
#     to_Conv("input", "", "", offset="(-2.5,0,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

#     # another invisible layer, for getting layer-size labels onto the input image
#     to_Conv("input_lab", "", ds[1][0], offset="(-3.7,-7.5,0)", to="(0,0,0)", height=0, depth=0, width=0 ),
#     to_Conv("input_lab", "", ds[1][1], offset="(-2.8,-7.1,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

#     # downsampling/encoding layers
#     to_Conv("conv0", ds[2][0], ds[2][1], offset="(-1,0,0)", to="(0,0,0)", height=my_trans(ds[2][1])*scal, depth=my_trans(ds[2][0])*scal, width=1 ),
#     to_connection( "input", "conv0"),
#     to_Pool("pool0", ds[3][0], ds[3][1], offset="(0.5,0,0)", to="(conv0-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1), 

#     to_Conv("conv1", ds[4][0], ds[4][1], offset="(1,0,0)", to="(pool0-east)",height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ),
#     to_connection( "pool0", "conv1"),
#     to_Pool("pool1", ds[5][0], ds[5][1], offset="(0.5,0,0)", to="(conv1-east)", height=my_trans(ds[5][1])*scal, depth=my_trans(ds[5][0])*scal, width=1),

#     # to_Dense("dense0", ds[6][0], ds[6][1], offset="(1,0,0)", to="(pool1-east)", height=my_trans(ds[6][1])*scal, depth=my_trans(ds[6][0])*scal, width=1), 
#     # to_connection("pool1", "dense0"),

    
#     # # flatten  + densep
#     # to_Dense("flatten", ds[7][0], ds[7][1], offset="(1,0,0)", to="(dense0-east)", height=my_trans(ds[7][1])*scal, depth=my_trans(ds[7][0])*scal, width=1 ),
#     # to_connection( "dense0", "flatten"),

#     # locs
#     to_input( 'locs_flip.png', to='(-3,-9.0,0)', width=4, height=1),
#     to_DensePurp("locs", "","", offset="(-0.5,-8.6,0)", to="(conv1-south)", height=0, depth=0, width=0),
#     to_DensePurp("locs2", ds[9][0], ds[9][1], offset="(-0.1,-8.1)", to="(pool1-south)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),
# #    to_connection( "locs", "locs2"),
#     to_Dense("pencil", "", "", offset="(1,0,0)", to="(pool1-east)", height=my_trans(ds[7][1])*scal, depth=my_trans(ds[7][0])*scal, width=1),
#     to_connection( "pool1", "pencil"),
#     to_DensePurp("eraser", ds[9][0], ds[7][1]+ds[9][1], offset="(-0.1,-0.2,0)", to="(pencil-south)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),    
#     to_connection( "locs2", "eraser"),

#     # invisible layer for some connections
#     to_Dense("invis1", "","", offset="(1,0,0)", to="(pencil-east)", height=0, depth=0, width=0),
#     to_connection( "pencil", "invis1"),    
#     to_Dense("invis2", "","", offset="(4,0,0)", to="(invis1-east)", height=0, depth=0, width=0),
            






    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

