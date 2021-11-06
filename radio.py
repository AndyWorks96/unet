
import six
import pandas as pd
from radiomics import featureextractor

def output_xlsx(dict_item,name):
    pf=pd.DataFrame(list(dict_item))
    xlsx_name=name+'.xlsx'
    xlsx_obj=pd.ExcelWriter(xlsx_name)
    pf.to_excel(xlsx_obj)
    xlsx_obj.save()




params = './data/pyradiomics_yaml/exampleMR_NoResampling.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

imageName='./data/t2Test/REN GUI LIAN^REN GUI/REN GUI LIAN^REN GUI_t2.nii.gz'
maskName = './data/t2Test/REN GUI LIAN^REN GUI/REN GUI LIAN^REN GUI_seg.nii.gz'
result = extractor.execute(imageName, maskName)

dict_item=result.items()
name = './data/t2Test/REN GUI LIAN^REN GUI'
output_xlsx(dict_item,name)


for key, val in six.iteritems(result):
    print("\t%s: %s" %(key, val))
