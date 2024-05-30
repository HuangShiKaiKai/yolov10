import os
import json
import cv2
import glob
import io
import numpy as np  # 导入numpy模块并指定别名np

# 1.标签路径
labelme_path = r"F:\subject"  # 原始json、bmp标注数据路径
saved_path = r"F:\subjectData"  # 保存路径
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# 2.获取待处理文件
files = glob.glob("%s/*.json" % (labelme_path))

# 3.读取标注信息并写入 xml
for json_filename in files:
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            json_file = json.load(f)

        # 图像名字
        img_name = json_filename.replace(".json", ".png")
        print("Processing image:", img_name)  # 调试信息

        # 读取图像
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError("Image file not found or the file is not a valid image.")

        # 获取图像尺寸
        height, width, channels = image.shape

        # xml名字
        xmlName = os.path.join(saved_path, os.path.basename(json_filename).replace(".json", ".xml"))

        with io.open(xmlName, "w", encoding="utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>png</folder>\n')
            xml.write('\t<filename>%s</filename>\n' % img_name)
            xml.write('\t<source>\n')
            xml.write('\t\t<database>hulan</database>\n')
            xml.write('\t</source>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>%d</width>\n' % width)
            xml.write('\t\t<height>%d</height>\n' % height)
            xml.write('\t\t<depth>%d</depth>\n' % channels)
            xml.write('\t</size>\n')
            xml.write('\t<segmented>0</segmented>\n')

            for multi in json_file["shapes"]:
                points = np.array(multi["points"])
                xmin, xmax = min(points[:, 0]), max(points[:, 0])
                ymin, ymax = min(points[:, 1]), max(points[:, 1])
                label = multi["label"]

                if xmax > xmin and ymax > ymin:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>%s</name>\n' % label)
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>%d</xmin>\n' % xmin)
                    xml.write('\t\t\t<ymin>%d</ymin>\n' % ymin)
                    xml.write('\t\t\t<xmax>%d</xmax>\n' % xmax)
                    xml.write('\t\t\t<ymax>%d</ymax>\n' % ymax)
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')
    except Exception as e:
        print("Error processing file {}: {}".format(json_filename, e))
        continue  # 如果处理过程中出现错误，跳过当前文件

print("Conversion completed.")