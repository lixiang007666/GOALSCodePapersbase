import paddlex as pdx
import glob
import os
import pandas as pd

save_dir = 'data/pplcnet'
os.makedirs(save_dir, exist_ok=True)

for fold_num in range(1, 11):
    model = pdx.load_model(f'models_cls/PPLCNet_Fold{fold_num}/epoch_50')

    test_list = sorted(glob.glob(f'GOALS2022-Validation{os.sep}GOALS2022-Validation{os.sep}Image{os.sep}*.png'))
    test_df = pd.DataFrame()

    for i in range(len(test_list)):
        result = model.predict(test_list[i])

        test_df.at[i, 'ImgName'] = str(test_list[i]).split(os.sep)[-1]
        test_df.at[i, 'GC_Pred'] = int(result[0]['category_id'])

    test_df[['ImgName']] = test_df[['ImgName']].astype(str)
    test_df[['GC_Pred']] = test_df[['GC_Pred']].astype(int)

    test_df.to_csv(os.path.join(save_dir, f'Classification_Results_Fold{fold_num}.csv'), index=False)

csv_path_list = glob.glob(os.path.join(save_dir, '*.csv'))
merge_list = []
for csv_path in csv_path_list:
    merge_list.append(pd.read_csv(csv_path, header=0)[['GC_Pred']])
df = pd.DataFrame()
df[['ImgName']] = pd.read_csv(csv_path_list[0], header=0)[['ImgName']]
df[['GC_Pred']] = pd.concat(merge_list, axis=1).mode(axis=1)

os.makedirs('results', exist_ok=True)
df.to_csv('results/Classification_Results.csv', index=False)
