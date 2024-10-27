import os

imagepath  = './goal/testing/Layer_Masks'

print(imagepath)

listimage = os.listdir(imagepath)
print(listimage)


for i,name in enumerate(listimage):

    newname = str(i+1).zfill(4) + '.png'

    os.rename(imagepath+'/'+ name,imagepath+'/'+newname)


#
#
#
# import os
#
# imagepath  = './Disc_Cup_Masks_0'
#
#
#
# listimage = os.listdir(imagepath)
# print(listimage)
#
#
# for i,name in enumerate(listimage):
#
#     newname = str(i+1).zfill(4) + '.bmp'
#
#     os.rename(imagepath+'/'+ name,imagepath+'/'+newname)