from radiomics import featureextractor
import os
import pandas as pd
import SimpleITK as sitk

basePath = 'D:/learning/lobectomyablation/radiomic/test/fuda'
folders = os.listdir(basePath)
# print(folders)

df = pd.DataFrame()
settings = {}

settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = [1, 1, 1]  # unit: mm
settings['interpolator'] = sitk.sitkNearestNeighbor
settings['normalize'] = True

# settings['geometryTolerance'] = 1e-1
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
# extractor = featureextractor.RadiomicsFeatureExtractor()
# extractor.enableImageTypes(Original={}, LoG={"sigma": [3.0, 4.0]}, Wavelet={})

# 指定使用 LoG 和 Wavelet 滤波器
extractor.enableImageTypeByName('LoG')
extractor.enableImageTypeByName('Wavelet')
# 选择所有特征
extractor.enableAllFeatures()



for folder in folders:
    files = os.listdir(os.path.join(basePath, folder))
    # print(files)
    for file in files:
        if file.endswith('image.nrrd'):
            imageFile = os.path.join(basePath, folder, file)
        if file.endswith('mask.nrrd'):
            maskFile = os.path.join(basePath, folder, file)
    # print(imageFile, maskFile)
    featureVector = extractor.execute(imageFile, maskFile)
#  把提取的影像组学特征储存在一个dataframe中，从字典中提取影像组学特征，并转秩
    df_new = pd.DataFrame.from_dict(featureVector.values()).T
#  把特征的名称作为新表格的列名
    df_new.columns = featureVector.keys()
    df_new.insert(0, 'imageFile', imageFile)
    df = pd.concat([df, df_new])
df.to_excel(os.path.join(basePath, 'results2.xlsx'), index=None)