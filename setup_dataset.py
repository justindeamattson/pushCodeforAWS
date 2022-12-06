from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

def main():
    # by running python dataset.py, you will create 3 csv files:
    # 1. train.csv
    # 2. val.csv
    # 3. test.csv
    # These csvs will be used as the data loaded in any of the experiments to ensure consistency in data splits
    ###############################################################################################
    # DATASET CONFIGURATIONS
    columns = ['team', 'skill', 'skill_type', 'skill_subtype', 'home_team_score', 'visiting_team_score', 'home_setter_position', 'visiting_setter_position', 'serving_team', "start_coordinate_x",	"start_coordinate_y",	"mid_coordinate_x",	"mid_coordinate_y",	"end_coordinate_x",	"end_coordinate_y",'home_touch', 'away_touch', 'point_won_by']
    data_path = 'export_our_test_value_FINAL.csv'
    ###############################################################################################
    # dataset = VolleyballDataset(data_path=data_path)
    print("Started processing data")
    dataset = pd.read_csv(data_path)
    #ataset.replace(to_replace= "nan", value=0)
    #print(dataset)
    print("Loaded data")

    #for i, row in dataset.iterrows():
        #print (dataset._get_value(i, 'mid_coordinate_x', takeable=False))
        #if dataset._get_value(i, 'mid_coordinate_x', takeable=False) is True:
          #  print("hi")
            #df.at[row,'team']=1
          #  dataset._set_value(i, 'mid_coordinate_x', 0)
    
    dataset = dataset.fillna(0)

    #dataset['mid_coordinate_x Column'] = dataset['mid_coordinate_x'].fillna(100)

    # Only use columns above
    dataset = dataset[columns]

    #if [dataset['start_coordinate_x'] == ""]:
     #   dataset['start_coordinate_x'] = 0

    #if [dataset['mid_coordinate_x'] == ""]:
    #    dataset['mid_coordinate_x'] = 0




    # point_won_by column should be 0 and 1
    dataset['point_won_by'] -= 1

    # Train, val, test
    train, test = train_test_split(dataset, random_state=0)
    val, test = train_test_split(test, test_size=0.5, random_state=0)

    train.to_csv("processed_data/train.csv", index=False)
    val.to_csv("processed_data/val.csv", index=False)
    test.to_csv("processed_data/test.csv", index=False)

    print("len(train):", len(train))
    print("len(val):", len(val))
    print("len(test):", len(test))
    print("Finished Processing Data")

main()