import os
import random
import shutil
 
#将图像文件夹，txt标签文件夹进行数据集的划分
 
# 定义原始图像和标签文件夹路径
image_folder = r'D:\python_test\data_aug\Augumentation\images'
label_folder = r'D:\python_test\data_aug\Augumentation\labels'
 
# 定义输出的根路径
output_root = r'D:\new_data_aug\new_pcb-3'
 
# 计算划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
 
#下面不用修改，上面按照需修改
 
# 创建 output_root 下的 images 和 labels 文件夹
output_images_folder = os.path.join(output_root, 'images')
output_labels_folder = os.path.join(output_root, 'labels')
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)
 
# 在 images 和 labels 文件夹下创建 train、val、test 子文件夹
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_images_folder, folder), exist_ok=True)
    os.makedirs(os.path.join(output_labels_folder, folder), exist_ok=True)
 
# 获取所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
 
# 筛选出有对应标签文件的图像
valid_image_files = []
for image in image_files:
    label_name = os.path.splitext(image)[0] + '.txt'
    label_path = os.path.join(label_folder, label_name)
    if os.path.exists(label_path):
        valid_image_files.append(image)
 
# 打乱有效图像文件列表
random.shuffle(valid_image_files)
 
 
# 计算每个数据集的数量
train_count = int(len(valid_image_files) * train_ratio)
val_count = int(len(valid_image_files) * val_ratio)
test_count = len(valid_image_files) - train_count - val_count
 
# 划分数据集
train_images = valid_image_files[:train_count]
val_images = valid_image_files[train_count:train_count + val_count]
test_images = valid_image_files[train_count + val_count:]
 
 
# 复制文件到对应的文件夹
def copy_files(images, image_dest_folder, label_dest_folder):
    for image in images:
        image_path = os.path.join(image_folder, image)
        label_name = os.path.splitext(image)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)
 
        # 复制图像文件
        shutil.copy2(image_path, image_dest_folder)
 
        # 复制标签文件
        if os.path.exists(label_path):
            shutil.copy2(label_path, label_dest_folder)
 
 
# 复制训练集文件
copy_files(train_images, os.path.join(output_images_folder, 'train'), os.path.join(output_labels_folder, 'train'))
 
# 复制验证集文件
copy_files(val_images, os.path.join(output_images_folder, 'val'), os.path.join(output_labels_folder, 'val'))
 
# 复制测试集文件
copy_files(test_images, os.path.join(output_images_folder, 'test'), os.path.join(output_labels_folder, 'test'))
 
print("数据集划分完成！")