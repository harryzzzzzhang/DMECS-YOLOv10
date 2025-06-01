from ultralytics import YOLOv10
import os

model_yaml_path = "./ultralytics/cfg/models/v10/yolov10n_DMECS.yaml" #
data_yaml_path = './ultralytics/cfg/datasets/a_color_dataset.yaml' # 完全年画

if __name__ == '__main__':
    model = YOLOv10(model_yaml_path)
    # model = YOLOv10(model_yaml_path)
    result = model.train(data=data_yaml_path, epochs=300, batch=20, name="your_test_name")
