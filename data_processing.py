import pandas as pd
                                           #处理系列(Series)、数据帧(DataFrame)、面板(Panel)此处处理DataFrame
from urllib.request import urlretrieve

def load_data(download=False):
    # download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",#要爬取的网站名
                                   "car.data.csv")
        print("Downloaded to car.data.csv")

    # use pandas to view the data structure 使用pandas查看数据结构
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.data.csv", names=col_names)
    return data
def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data, prefix=data.columns)
if __name__ == "__main__":#__name__ 是当前模块名，
                          # 当模块被直接运行时模块名为 __main__模块被直接运行时，以下代码块将被运行，
                          # 当模块是被导入时，代码块不被运行
    data = load_data(download=False)
    new_data = convert2onehot(data)

    print(data.head())
    print("\nNum of data: ", len(data), "\n")
    # view data values查看视图的数据的值
    for name in data.keys():
        print(name, pd.unique(data[name]))
    print("\n", new_data.head(2))
    new_data.to_csv("car_onehot.csv", index=False) #数据输出，写入到car_onehot.cav，此时没有索引值