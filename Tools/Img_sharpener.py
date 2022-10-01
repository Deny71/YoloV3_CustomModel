import cv2
import numpy as np
import glob
import os
import shutil

# Creating our sharpening filter
kernelSize = 3
centerValue = kernelSize**2
centralIndex = int(kernelSize/2)
kernel = np.full((kernelSize, kernelSize), -1, dtype=int)
kernel[centralIndex][centralIndex] = centerValue

Path = os.path.dirname(os.path.realpath(__file__))
Path = os.path.split(Path)
PathTrimmed = Path[0] + "\\OIDv4_ToolKit\\OID\\Dataset\\test_sharp\\Swan\\"
#os.mkdir('sharpTest')

for filePath in glob.glob(PathTrimmed + "*.*"):
    if filePath.endswith(".jpg"):
        print(filePath)

        img = cv2.imread(filePath)
        modFilename = os.path.split(filePath)
        sharpening = cv2.filter2D(img, -1, kernel)

        resultPath = os.path.split(PathTrimmed)
        filenameSharpening = PathTrimmed + "\\sharpened11x11\\" + modFilename[1]
        cv2.imwrite(filenameSharpening, sharpening)

    elif filePath.endswith(".xml"):
        print(filePath)
        modFilename = os.path.split(filePath)
        dst = PathTrimmed + "\\sharpened11x11\\" + modFilename[1]
        shutil.copy2(filePath, dst)

print("Image sharpening done!")