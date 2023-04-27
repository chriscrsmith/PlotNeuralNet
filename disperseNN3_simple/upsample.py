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
      [150,150], # upsample1
      [(np.sqrt(150)-np.sqrt(45))**2, (np.sqrt(5292)+np.sqrt(4))**2], # padding2
      [500,500], # upsample2
      [(np.sqrt(500)-np.sqrt(45))**2, (np.sqrt(5292)+np.sqrt(4))**2], # padding3                                                                                                                     
#      [500,500], # upsample3
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
    to_DenseOrange("dense0", ds[2][0], ds[2][1], offset="(1,0,0)", to="(feature_block1-east)", height=my_trans(ds[2][1])*scal, depth=my_trans(ds[2][0])*scal, width=1),
    to_connection("feature_block1", "dense0"),                                                                                                                       

    # upsample
    to_DenseOrange("upsample1", ds[3][0], ds[3][1], offset="(1,0,0)", to="(dense0-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),
    to_connection("dense0", "upsample1"),

    ########################### these offsets work great for width=100
    # # feature block
    # to_Dense("feature_block2", "", "", offset="(1,0,0)", to="(upsample1-east)", height=my_trans(ds[0][1])*scal, depth=my_trans(ds[0][0])*scal, width=1 ),                            
    # to_connection("upsample1", "feature_block2"),                                                                                                                                    
    # to_DensePurp("locs2", ds[1][0], ds[0][1]+ds[1][1], offset="(-0.1,-0.2,0)", to="(feature_block2-south)", height=my_trans(ds[1][1])*scal, depth=my_trans(ds[1][0])*scal, width=1),
    # to_Dense("padding2", "", "", offset="(-0.475, -0.475, -1.75)", to="(feature_block2-east)", height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ), # SUPER painstaking
    # to_Dense("concat2", ds[3][0], ds[3][1], offset="(-0.475, -8.75, 0)", to="(padding2-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),
    ##########################

    # feature block                                                                                                                                                                         
    to_Dense("feature_block2", "", "", offset="(1.25,0,0)", to="(upsample1-east)", height=my_trans(ds[0][1])*scal, depth=my_trans(ds[0][0])*scal, width=1 ),                                   
    to_connection("upsample1", "feature_block2"),                                                                                                                                           
    to_DensePurp("locs2", ds[1][0], ds[0][1]+ds[1][1], offset="(-0.1,-0.2,0)", to="(feature_block2-south)", height=my_trans(ds[1][1])*scal, depth=my_trans(ds[1][0])*scal, width=1),        
    to_DenseGrey("padding2", "", "", offset="(-0.475, -0.475, -1.925)", to="(feature_block2-east)", height=my_trans(ds[4][1])*scal, depth=my_trans(ds[4][0])*scal, width=1 ), # SUPER painstaking
    to_DenseOrange("concat2", ds[3][0], ds[3][1], offset="(-0.48, -9, -0.0825)", to="(padding2-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),                     
    
    # map 2                                                                                                                                                                                 
    to_DenseOrange("dense2", ds[3][0], ds[3][1], offset="(1.5,0,0)", to="(feature_block2-east)", height=my_trans(ds[3][1])*scal, depth=my_trans(ds[3][0])*scal, width=1),
    to_connection("feature_block2", "dense2"),

    # upsample                                                                                                                                                                              
    to_DenseOrange("upsample2", ds[5][0], ds[5][1], offset="(1.75,0,0)", to="(dense2-east)", height=my_trans(ds[5][1])*scal, depth=my_trans(ds[5][0])*scal, width=1),
    to_connection("dense2", "upsample2"),

    # feature block                                                                                                                                                                         
    to_Dense("feature_block3", "", "", offset="(1.5,0,0)", to="(upsample2-east)", height=my_trans(ds[0][1])*scal, depth=my_trans(ds[0][0])*scal, width=1 ),
    to_connection("upsample2", "feature_block3"),
    to_DensePurp("locs3", ds[1][0], ds[0][1]+ds[1][1], offset="(-0.1,-0.2,0)", to="(feature_block3-south)", height=my_trans(ds[1][1])*scal, depth=my_trans(ds[1][0])*scal, width=1),
    to_DenseGrey("padding3", "", "", offset="(-0.475, -0.475, -2.95)", to="(feature_block3-east)", height=my_trans(ds[6][1])*scal, depth=my_trans(ds[6][0])*scal, width=1 ), # SUPER painstaking
    to_DenseOrange("concat3", ds[5][0], ds[5][1], offset="(-0.475, -10, -0.05)", to="(padding3-east)", height=my_trans(ds[5][1])*scal, depth=my_trans(ds[5][0])*scal, width=1),

    # invisible layer for drawing connection
    to_Dense("invis1", "","", offset="(2,0,0)", to="(feature_block3-east)", height=0, depth=0, width=0),
    to_connection("feature_block3", "invis1"),
   
    
    
    
    

            






    
    to_end(),

    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()


