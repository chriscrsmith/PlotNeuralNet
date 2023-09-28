import sys
sys.path.append('../')
from my_layers import *
import numpy as np



def my_trans(x):
    #return x**(1/2) 
    return x**(1/3)

scal = 1 # multiply this by the transformed layer sizes
pairs = 45
width = 10

# dims for genotpe input layers
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
      [1, 128], # 10
      [45, 1], # 11
      [10*10*45,128, 128], # 12
#      [45, 128], # 13
      ]

ds2 = [[pairs*width**2,128], # 0 
      [pairs*width**2,64], # 1    
      [width**2,64], # 2   
      [width**2,2], # 3    
       [width,width], # 4   
#       [108,49], # 5    
#       [1,5292], # 6    
#       [1,128], # 7     
#       [None, None], # 8
#       [1,1], # 9       
#       [1, 128], # 10   
#       [45, 1], # 11    
#       [1, 3250], # 12  
#       [45, 128], # 13  
# 
       ]


# define the layers
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    ############################# GENOTYPE INPUT #############################################
    
    # # input
    # to_input( 'genos_flip.png', width=1, height=14), # might need jpg...

    # invisible layer, only for drawing a connection to the input image
    to_Conv("input", "", "", offset="(-1,0,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

    # # another invisible layer, for getting layer-size labels onto the input image
    # to_Conv("input_lab", "", ds[1][0], offset="(-3.7,-7.5,0)", to="(0,0,0)", height=0, depth=0, width=0 ),
    # to_Conv("input_lab", "", ds[1][1], offset="(-2.8,-7.1,0)", to="(0,0,0)", height=0, depth=0, width=0 ),

    # downsampling/encoding layers
    to_Conv("conv0", "", "", offset="(-0.5,0,0)", to="(0,0,0)", height=my_trans(ds[2][1])*scal, depth=my_trans(ds[2][0])*scal, width=1 ),
    to_connection( "input", "conv0"),
    to_Pool("pool0", "", "", offset="(0.25,0,0)", to="(conv0-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1), 
    to_Conv("conv1", "", "", offset="(0.5,0,0)", to="(pool0-east)",height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ),
    to_connection( "pool0", "conv1"),
    to_Pool("pool1", "", "", offset="(0.25,0,0)", to="(conv1-east)", height=my_trans(ds[5][1])*scal, depth=my_trans(ds[5][0])*scal, width=1),

    # # locs input
    # to_input( 'locs_flip.png', to='(4.75,-5.5,0)', width=4, height=1),
    # to_DensePurp("locs", "","", offset="(3,-5.5)", to="(pool1-south)", height=0, depth=0, width=0), # invisible, for connecting locs-input to first 1x1 square
    # to_DensePurp("locs2", ds[9][0], ds[9][1], offset="(3,-6.65,0)", to="(pool1-south)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1), # 1x1 square
    # to_connection( "locs", "locs2"),

    # feature vector
    to_Dense("pencil", "","", offset="(0.5,0,0)", to="(pool1-east)", height=my_trans(ds[6][1])*scal, depth=my_trans(ds[6][0])*scal, width=1),
    to_connection( "pool1", "pencil"),
    # to_DensePurp("eraser", ds[9][0], ds[6][1]+ds[9][1], offset="(-0.1,-0.1,0)", to="(pencil-south)", height=my_trans(ds[9][1])*scal, depth=my_trans(ds[9][0])*scal, width=1),
    # to_connection( "locs2", "eraser"),

    # dense layers
    to_Dense("dense0", "", "", offset="(0.25,0,0)", to="(pencil-east)", height=my_trans(ds[7][1])*scal, depth=my_trans(ds[7][0])*scal, width=1),
    to_connection("pencil", "dense0"),

    # invisible layer for some connections (leaving a blank space for the recycle symbol)
    to_Dense("invis1", "","", offset="(0.5,0,0)", to="(dense0-east)", height=0, depth=0, width=0),
    to_connection( "dense0", "invis1"),    
    to_Dense("invis2", "","", offset="(1.5,0,0)", to="(invis1-east)", height=0, depth=0, width=0), # first dim controls width of the gap
            
    # feature block
    to_Dense("feature_block", "","", offset="(1.5,0,0)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_Dense("feature_block2", "","",  offset="(1.5,0,0.5)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_Dense("feature_block3", "","",  offset="(1.5,0,1)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_Dense("feature_block4", "","",  offset="(1.5,0,1.5)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_Dense("feature_block5", "","",  offset="(1.5,0,2)", to="(invis2-east)", height=my_trans(ds[10][1])*scal, depth=my_trans(ds[10][0])*scal, width=1 ),
    to_Dense("invis3", "","", offset="(0.75,0,0)", to="(invis2-east)", height=0, depth=0, width=0),
    to_connection( "invis2", "invis3"),

    # dense 
    to_Dense("dense1", "", "", offset="(0.5,0,0)", to="(feature_block-east)", height=my_trans(ds[12][0])*scal, depth=my_trans(ds[12][1])*scal, width=1),
    to_connection( "feature_block", "dense1"),
    # to_Dense("dense2", ds[13][0], ds[13][1], offset="(0.75,0,0)", to="(dense1-east)", height=my_trans(ds[13][1])*scal, depth=my_trans(ds[13][0])*scal, width=1),
    # to_connection( "dense1", "dense2"),

    # # output
    # to_Dense("output", 1, 1, offset="(0.75,0,0)", to="(dense1-east)", height=my_trans(1)*scal, depth=my_trans(1)*scal, width=1),
    # to_connection( "dense1", "output"),





    
    ########################## LOCS INPUT #########################################

    # DENSE
    to_Dense("dense2", "", "", offset="(0,-3.5,0)", to="(feature_block3-south)", height=my_trans(ds2[0][0])*scal, depth=my_trans(ds2[0][1])*scal, width=1),
    #to_connection("pencil", "dense0"),

    # draw invisible layer to the LEFT of the dense layer
    to_Conv("invis4", "", "", offset="(-0.75,0,0)", to="(dense2-west)", height=0, depth=0, width=0 ),
    to_connection( "invis4", "dense2"),
    
    # DENSE
    to_Dense("dense3", "", "", offset="(0.75,0,0)", to="(dense2-east)", height=my_trans(ds2[0][0])*scal, depth=my_trans(ds2[0][1])*scal, width=1),
    to_connection("dense2", "dense3"),                                                                                                                                                                 




    
    ########################## COMBINED ###########################################
    to_Dense("dense4", "", "", offset="(0.75,2,0)", to="(dense3-east)", height=my_trans(ds2[0][0])*scal, depth=my_trans(ds2[0][1])*scal, width=1),
    to_connection("dense1", "dense4"),
    to_connection("dense3", "dense4"),

    to_Dense("dense5", "", "", offset="(0.25,0,0)", to="(dense4-east)", height=my_trans(ds2[1][0])*scal, depth=my_trans(ds2[1][1])*scal, width=1),
    to_connection("dense4", "dense5"),

    to_Dense("dense6", "", "", offset="(0.25,0,0)", to="(dense5-east)", height=my_trans(ds2[2][0])*scal, depth=my_trans(ds2[2][1])*scal, width=1),
    to_connection("dense5", "dense6"),

    to_Dense("dense7", "", "", offset="(0.25,0,0)", to="(dense6-east)", height=my_trans(ds2[2][0])*scal, depth=my_trans(ds2[2][1])*scal, width=1),
    to_connection("dense6", "dense7"),

    to_Dense("dense8", "", "", offset="(0.25,0,0)", to="(dense7-east)", height=my_trans(ds2[2][0])*scal, depth=my_trans(ds2[2][1])*scal, width=1),
    to_connection("dense7", "dense8"),

    to_Dense("dense9", "", "", offset="(0.25,0,0)", to="(dense8-east)", height=my_trans(ds2[3][0])*scal, depth=my_trans(ds2[3][1])*scal, width=1),
    to_connection("dense8", "dense9"),

    to_Dense("dense10", "", "", offset="(0.25,0,0)", to="(dense9-east)", height=my_trans(ds2[4][0])*scal, depth=my_trans(ds2[4][1])*scal, width=1),
    to_connection("dense9", "dense10"),


    










    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

