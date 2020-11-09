#..................for real_image_data..................................
deepfake_data_ori=pd.read_csv('/content/drive/My Drive/deepfake_data/deepfake_data.csv')
deepfake_data_ori.head()

train_ori_label=deepfake_data_ori.drop(['ori_vid'], axis = 1)
train_ori_label=train_ori_label.reset_index(drop=True)
train_ori_label["label"] =train_ori_label["label"].replace({ 0: 'real'})
train_ori_label.head()
print(train_ori_label.shape)

#.......................for fake_image_data....................
deepfake_data_fake=pd.read_csv('/content/drive/My Drive/deepfake_data/deepfake_data_1_2.csv')
deepfake_data_fake.head()

train_fake_label=deepfake_data_fake.drop(['ori_vid'], axis = 1)
train_fake_label=train_fake_label.reset_index(drop=True)
train_fake_label["label"] =train_fake_label["label"].replace({ 1: 'fake'})
print(train_fake_label.shape[0])
print(train_fake_label.head())

#....................for real and fake images data....................................

train_data=deepfake_data_ori.append(deepfake_data_fake)
train_data=train_data.drop(['ori_vid'], axis = 1)
train_data=train_data.reset_index(drop=True)
train_data["label"] =train_data["label"].replace({0:'real', 1: 'fake'})
print(train_data.shape[0])
print(train_data.head())
#train_data.tail(10)

#...............randomly split real and fake images data......................

from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(train_data, test_size=0.20, random_state=42)
