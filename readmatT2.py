import scipy.io as io

matT2 = io.loadmat("/Users/andyworks/Desktop/test/matlab.mat")
mask = matT2['rectalMask']
npimage = mask.transpose((2, 0, 1))


slice = npimage[32]




print(npimage[5])