![roadmap](https://cvdl.cupoy.com/images/learnWithCoachLogin.png)  

* [排名賽](https://cvdl.cupoy.com/ranking/homeworkrank)   
* [個人主頁](https://cvdl.cupoy.com/participator/84F13837/questions)  

## 1. 基礎影像處理
> 學習影像處理基礎，並熟悉 OpenCV 寫作方式以及如何前處理

1. [OpenCV 簡介 + 顯示圖片](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day001_read_image_HW.ipynb) `入門電腦視覺領域的重要套件: OpenCV`
2. [Color presentation 介紹 (RGB, LAB, HSV)](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day002_change_color_space_HW.ipynb) `淺談圖片不同的表示方式`
3. [顏色相關的預處理 (改變亮度, 色差)](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day003_color_spave_op_HW.ipynb) `透過圖片不同的表示方式實作修圖效果`
4. [以圖片為例做矩陣操作 (翻轉, 縮放, 平移)](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day004_geometric_transform_HW.ipynb) `淺談基礎的幾合變換: 翻轉 / 縮放 / 平移`
5. [透過 OpenCV 做圖並顯示 (長方形, 圓形, 直線, 填色)](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day005_draw_HW.ipynb) `實作 OpenCV 的小畫家功能`
6. [affine transformation 概念與實作](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day006_affine_HW.ipynb) `仿射轉換的入門與實作: affine transform`
7. perspective transformation 概念與實作 `視角轉換的入門與實作: perspective transform`
8. [Filter 操作 (Sobel edge detect, Gaussian Blur)](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day008_sobel_gaussian_blur_HW.ipynb) `初探邊緣檢測與模糊圖片操作: 了解 filter 的運用`
9. SIFT 介紹與實作 (feature extractor) `SIFT: 介紹與實作經典的傳統特徵`
10. SIFT 其他應用 (keypoint matching) `SIFT 案例分享: 特徵配對`

## 2. 電腦視覺深度學習基礎
> 打好卷積神經網路的概念，並了解 CNN 各種代表性的經典模型
11. [CNN分類器架構：卷積層](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day011_CNN-%E8%A8%88%E7%AE%97%E5%8F%83%E6%95%B8%E9%87%8F_HW.ipynb) `卷積是CNN的核心，了解卷積如何運行 就能幫助我們理解CNN的原理`
12. [CNN分類器架構：步長、填充](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day012_Strides%2Band%2BPadding_HW.ipynb) `填充與步長是CNN中常見的超參數， 了解如何設置能幫助我們架構一個CNN Model`
13. [CNN分類器架構：池化層、全連接層](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day013_%E6%B1%A0%E5%8C%96%E3%80%81%E5%85%A8%E9%80%A3%E6%8E%A5%E5%B1%A4_HW.ipynb) `池化層時常出現於CNN結構中，而FC層則會接在模型輸出端， 了解如兩者用途能幫助我們架構一個CNN Model`
14. [CNN分類器架構：Batch Normalization](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day014_Batch%2BNormalization_HW.ipynb) `Batch Normalization出現在各種架構中， 了解BN層能解決怎樣的問題是我們本章的重點`
15. [訓練一個CNN分類器：Cifar10為例](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day015_Cifar_HW.ipynb) `綜合上述CNN基本觀念， 我們如何結合這些觀念打造一個CNN 模型`
16. [如何使用Data Augmentation](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day016_Image%2BAugmentation_HW.ipynb) `訓練模型時常常會遇到資料不足的時候，適當的使用Image Augmentation能提升模型的泛化性`
17. AlexNet `綜合之前所學的CNN觀念，認識第一個引領影像研究方向朝向深度學習的模型`
18. [VGG16 and 19](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day018_Vgg16_HW.ipynb) `模型繼續進化，認識簡單卻又不差的CNN模型`
19. [InceptionV1-V3](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day019_Inception_HW.ipynb) `Inception module提供大家不同於以往的思考方式，將模型的參數量減少，效能卻提升了許多`
20. [ResNetV1-V2、InceptionV4、Inception-ResNet](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day020_Classic%2BCNN-ResNet%E3%80%81InceptionV4%E3%80%81Inception-ResNet_HW.ipynb) `首次超越人類分類正確率的模型，Residual module也影響了後來許多的模型架構`
21. [Transfer learning](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day021_Transfer%2BLearning_HW.ipynb) `學習如何利用前人的知識輔助自己訓練與跨領域學習的方法`
22. [Breaking Captchas with a CNN](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day022_Captcha_HW.ipynb) `了解如何使用CNN+CTC判定不定長度字串`

## 3. CNN 應用案例學習
> 學習目前最常使用的 CNN 應用案例：YOLO 物件偵測實務完全上手
### Object detection
23. Object detection原理 `了解Object Detection出現的目的與基本設計原理`
24. Object detection基本介紹、演進 `了解Object Detection一路發展下來，是如何演進與進步`
25. [Region Proposal、IOU概念](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day025_IOU_HW.ipynb) `IOU是貫穿Object Detection的一個重要觀念，了解如何計算IOU對了解Object Detection中許多重要步驟會很有幫助`
26. [RPN架構介紹](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day026_RPN_HW.ipynb) `RPN是Faster RCNN成功加速的關鍵，了解RPN便能深入認識Faster RCNN`
27. [Bounding Box Regression原理](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day027_BBOX%2BRegression_HW.ipynb) `所有的Object Detection模型都需要做Bounding Box的Regression，了解其是如何運作的能幫助我們更認識Object Detection`
28. [Non-Maximum Suppression (NMS)原理](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day028_NMS_HW.ipynb) `所有的Object Detection模型都有Non Maximum Suppression的操作，了解其是如何運作的能幫助我們更認識Object Detection`
29. [程式導讀、實作](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day029_Build_SSD_VGG.ipynb) `了解如何搭建一個SSD模型`   

### YOLO
32.  [YOLO 簡介及算法理解](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day032_yolo_prediction_HW.ipynb) `了解 YOLO 的基本原理`
33. [YOLO 細節理解 - 網路輸出的後處理](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day033_YOLO%2B%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3%2B-%2B%E7%B6%B2%E8%B7%AF%E8%BC%B8%E5%87%BA%E7%9A%84%E5%BE%8C%E8%99%95%E7%90%86_HW.ipynb) `理解網路輸出的後處理，執行nms的過程`
34. [YOLO 細節理解 - 損失函數](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day034.YOLO%2B%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3%2B-%2B%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8_HW.ipynb) `認識YOLO損失函數設計架構與定義`
35. [YOLO 細節理解 - 損失函數程式碼解讀](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day035_yolo_loss.ipynb) `講解YOLO損失函數程式碼`
36. [YOLO 細節理解 - 網路架構](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day036_YOLO%2B%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3-%E7%B6%B2%E7%B5%A1%E6%9E%B6%E6%A7%8B_HW.ipynb) `了解YOLO網絡架構的設計與原理`
37. [YOLO 細節理解 - 網路架構程式碼解讀](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day037_YOLO_network.ipynb) `講解YOLO網絡架構程式碼`
38. YOLO 演進 `簡單了解 YOLO 的演進`
39. [使用 YOLOv3 偵測圖片及影片中的物件](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day039_yolov3_keras_HW.ipynb) `了解如何基於現有的 YOLOv3 程式碼客制化自己的需求`
40. [更快的檢測模型 - tiny YOLOv3](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day040_tiny_yolov3_keras_HW.ipynb) `了解如何使用 tiny YOLOv3 來滿足對檢測速度的需求`
41. [訓練 YOLOv3](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day041_train_yolov3_HW.ipynb) `了解如何訓練 YOLOv3 檢測模型`

## 4. 電腦視覺深度學習實戰
> 人臉關鍵點檢測及其應用
42. [人臉關鍵點-資料結構簡介](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day042_explore_facial_keypoint_data_HW.ipynb) `探索 kaggle 臉部關鍵點資料`
43. [人臉關鍵點-檢測網路架構](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day043_define_network_HW.ipynb) `學習用 keras 定義人臉關鍵點檢測的網路架構`
44. [訓練人臉關鍵點檢測網路](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day044_train_facial_keypoint_HW.ipynb) `體會訓練人臉關鍵點檢測網路的過程`
45. [人臉關鍵點應用](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day045_facial_keypoint_application.ipynb) `體驗人臉關鍵點的應用 - 人臉濾鏡`
46. [Mobilenet](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day046_MobileNet_HW.ipynb) `輕量化模型簡介 (MobileNet)`
47. [Ｍobilenetv2](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day047_MobileNetv2_HW.ipynb) `MobileNet v2 簡介`
48. [Tensorflow Object Detection API](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day048_tensorflow_object_detection_api_training.ipynb) `Tensorflow Object Detection API使用方式`


## 5. 期末專題
> 由您親手實作期末專題，驗收學習成效，陪跑專家提供專題解說，解決各種實作疑難雜症   

[Final Project](https://github.com/ZivChien/1st-DL-CVMarathon/blob/master/homework/Day049_final.ipynb)
