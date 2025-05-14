import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';
import 'models.dart';
import 'dart:math';

/// OnnxDetector類別用於載入ONNX模型並進行物件檢測
class OnnxDetector {
  OrtSession? _session;
  final int _modelInputSize = 320;
  late final Float32List _inputBuffer = Float32List(
    1 * 3 * _modelInputSize * _modelInputSize,
  );

  /// COCO資料集的80個類別
  final List<String> _cocoClasses = [
    'nail',
    'fifty_coin',
    'five_coin',
    'one_coin',
    'ten_coin',
  ];

  /// 載入ONNX模型
  Future<void> loadModel() async {
    try {
      OrtEnv.instance.init();
      const assetFileName = 'assets/models/yolo11n320nailsdete1229.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(
        bytes,
        OrtSessionOptions()
          ..setIntraOpNumThreads(4)
          ..setInterOpNumThreads(2),
      );
      print('ONNX模型載入成功');
    } catch (e) {
      print('ONNX模型載入失敗: $e');
    }
  }

  /// 對相機圖像進行物件檢測
  Future<List<Detection>> detect(CameraImage image) async {
    try {
      final originalWidth = image.width.toDouble();
      final originalHeight = image.height.toDouble();

      /// 將相機圖像轉換為RGB格式
      img.Image rgbImage;
      if (image.format.group == ImageFormatGroup.yuv420) {
        rgbImage = _convertYUV420ToRGB(image);
        // print('YUV420格式圖像');
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        rgbImage = _convertBGRA8888ToRGB(image);
        // print('BGRA8888格式圖像');
      } else {
        throw Exception('Unsupported image format');
      }

      // 計算輸入圖像的最大邊長
      final maxDim = max(originalWidth, originalHeight);
      // 計算縮放因子 (與C++中的x_factor和y_factor相同)
      final xFactor = originalWidth / _modelInputSize;
      final yFactor = originalHeight / _modelInputSize;

      // 將圖像縮放到模型輸入尺寸
      final resizedImage = img.copyResize(
        rgbImage,
        width: _modelInputSize,
        height: _modelInputSize,
        interpolation: img.Interpolation.linear,
      );

      _fillInputBuffer(resizedImage);

      /// 執行模型推理
      final outputs = await _runInference();

      /// 處理模型輸出並生成檢測結果
      final detections = _processOutput(
        outputs,
        originalWidth,
        originalHeight,
        xFactor,
        yFactor,
      );

      return detections;
    } catch (e) {
      print('物件檢測失敗: $e');
      return [];
    }
  }

  /// 將YUV420格式圖像轉換為RGB格式
  img.Image _convertYUV420ToRGB(CameraImage image) {
    final imageWidth = image.width;
    final imageHeight = image.height;
    // print('image: ${image.width}, ${image.height}');
    final yBuffer = image.planes[0].bytes;
    final uBuffer = image.planes[1].bytes;
    final vBuffer = image.planes[2].bytes;
    final int yRowStride = image.planes[0].bytesPerRow;
    final int yPixelStride = image.planes[0].bytesPerPixel!;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    // Create the image with swapped width and height to account for rotation
    final output = img.Image(width: imageHeight, height: imageWidth);
    for (int h = 0; h < imageHeight; h++) {
      int uvh = (h / 2).floor();
      for (int w = 0; w < imageWidth; w++) {
        int uvw = (w / 2).floor();
        final yIndex = (h * yRowStride) + (w * yPixelStride);
        final int y = yBuffer[yIndex];
        final int uvIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);
        final int u = uBuffer[uvIndex];
        final int v = vBuffer[uvIndex];
        int r = (y + v * 1436 / 1024 - 179).round();
        int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
        int b = (y + u * 1814 / 1024 - 227).round();
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);
        // Set the pixel with rotated coordinates
        output.setPixelRgb(imageHeight - h - 1, w, r, g, b);
      }
    }
    // print('output: ${output.width}, ${output.height}');
    // img.resize(output, width: imageWidth, height: imageHeight);
    return output;
  }

  /// 將BGRA8888格式圖像轉換為RGB格式
  img.Image _convertBGRA8888ToRGB(CameraImage image) {
    final bytes = image.planes[0].bytes;
    final originalImage = img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
    final rgbImage = img.Image(width: image.width, height: image.height);
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = originalImage.getPixel(x, y);
        final r = pixel.r.toInt();
        final g = pixel.g.toInt();
        final b = pixel.b.toInt();
        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }
    return rgbImage;
  }

  /// 填充模型輸入緩衝區
  void _fillInputBuffer(img.Image image) {
    int pixelIndex = 0;
    final int channelSize = _modelInputSize * _modelInputSize;
    for (int y = 0; y < _modelInputSize; y++) {
      for (int x = 0; x < _modelInputSize; x++) {
        final pixel = image.getPixel(x, y);
        _inputBuffer[pixelIndex] = (pixel.r.toDouble() / 255.0);
        _inputBuffer[pixelIndex + channelSize] = (pixel.g.toDouble() / 255.0);
        _inputBuffer[pixelIndex + 2 * channelSize] =
            (pixel.b.toDouble() / 255.0);
        pixelIndex++;
      }
    }
  }

  /// 執行ONNX模型推理
  Future<List<OrtValue?>> _runInference() async {
    try {
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        _inputBuffer,
        [1, 3, _modelInputSize, _modelInputSize],
      );
      final runOptions = OrtRunOptions();
      final inputs = {'images': inputTensor};
      final outputs = await _session?.runAsync(runOptions, inputs) ?? [];
      inputTensor.release();
      runOptions.release();
      return outputs;
    } catch (e) {
      print('模型推理失敗: $e');
      return [];
    }
  }

  List<Detection> _processOutput(
    List<OrtValue?> outputs,
    double originalWidth,
    double originalHeight,
    double xFactor,
    double yFactor,
  ) {
    try {
      /// 檢查輸出是否有效
      if (outputs.isEmpty || outputs[0] == null) {
        return [];
      }
      final outputTensor = outputs[0] as OrtValueTensor;
      final data = outputTensor.value as List<dynamic>;
      if (data.isEmpty || data[0].isEmpty) {
        return [];
      }

      final dataShape = _getListShape(data);
      final List<Detection> candidateDetections = [];

      // 處理檢測輸出，與C++中的循環對應
      if (dataShape.length >= 2 && dataShape[0] == 1) {
        final batch0 = data[0];
        if (batch0 is List) {
          int numClasses = 80;
          int numBBoxValues = 4;
          int numDetections = batch0[0].length;

          for (int i = 0; i < numDetections; i++) {
            try {
              // 獲取類別分數並找到最高分數的類別
              List<double> classScores = [];
              for (int j = 0; j < numClasses; j++) {
                int scoreIndex = numBBoxValues + j;
                if (scoreIndex < batch0.length) {
                  double score = (batch0[scoreIndex][i] as num).toDouble();
                  classScores.add(score);
                }
              }

              int maxClassId = 0;
              double maxScore = 0.0;
              for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxScore) {
                  maxScore = classScores[j];
                  maxClassId = j;
                }
              }

              final confidenceThreshold = 0.7;
              if (maxScore >= confidenceThreshold) {
                final cx = (batch0[0][i] as num).toDouble();
                final cy = (batch0[1][i] as num).toDouble();
                final w = (batch0[2][i] as num).toDouble();
                final h = (batch0[3][i] as num).toDouble();

                // 將模型輸出轉換為原始圖像上的座標，模仿C++代碼中的計算方式
                double x = ((cx - 0.5 * w) * xFactor);
                double y = ((cy - 0.5 * h) * yFactor);
                double width = (w * xFactor);
                double height = (h * yFactor);

                // 計算邊界框的左上角和右下角座標
                final left = max(0.0, x.toDouble());
                final top = max(0.0, y.toDouble());
                final right = min(originalWidth, (x + width).toDouble());
                final bottom = min(originalHeight, (y + height).toDouble());

                // 確保邊界框有效
                if (right <= left || bottom <= top) {
                  continue;
                }

                // 創建檢測對象
                candidateDetections.add(
                  Detection(
                    boundingBox: [left, top, right, bottom],
                    confidence: maxScore,
                    classId: maxClassId,
                    className: _cocoClasses[maxClassId],
                  ),
                );
              }
            } catch (e) {
              print('處理檢測框 $i 失敗: $e');
            }
          }
        }
      }

      // 排序候選檢測並應用NMS (與C++代碼中的NMSBoxes對應)
      if (candidateDetections.isEmpty) {
        return [];
      }

      candidateDetections.sort((a, b) => b.confidence.compareTo(a.confidence));
      final nmsDetections = _applyNMS(candidateDetections);
      return nmsDetections;
    } catch (e) {
      print('處理模型輸出失敗: $e');
      return [];
    } finally {
      for (var element in outputs) {
        element?.release();
      }
    }
  }

  /// 獲取輸出數據的形狀
  List<int> _getListShape(dynamic list) {
    final shape = <int>[];
    dynamic current = list;
    while (current is List && current.isNotEmpty) {
      shape.add(current.length);
      current = current[0];
    }
    return shape;
  }

  /// 應用非極大值抑制（NMS）過濾重疊檢測框
  List<Detection> _applyNMS(List<Detection> boxes) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<Detection> selected = [];
    final Set<int> suppressed = {};
    final nmsThreshold = 0.45;

    for (int i = 0; i < boxes.length; i++) {
      if (suppressed.contains(i)) continue;
      selected.add(boxes[i]);
      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed.contains(j)) continue;
        if (boxes[i].classId != boxes[j].classId) continue;
        final double iou = _calculateIOU(
          boxes[i].boundingBox,
          boxes[j].boundingBox,
        );
        if (iou > nmsThreshold) {
          suppressed.add(j);
        }
      }
    }
    return selected;
  }

  /// 計算兩個邊界框的交並比（IOU）
  double _calculateIOU(List<double> boxA, List<double> boxB) {
    final x1A = boxA[0];
    final y1A = boxA[1];
    final x2A = boxA[2];
    final y2A = boxA[3];
    final x1B = boxB[0];
    final y1B = boxB[1];
    final x2B = boxB[2];
    final y2B = boxB[3];
    final xA = max(x1A, x1B);
    final yA = max(y1A, y1B);
    final xB = min(x2A, x2B);
    final yB = min(y2A, y2B);
    if (xB <= xA || yB <= yA) return 0.0;
    final interArea = (xB - xA) * (yB - yA);
    final boxAArea = (x2A - x1A) * (y2A - y1A);
    final boxBArea = (x2B - x1B) * (y2B - y1B);
    final unionArea = boxAArea + boxBArea - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0;
  }

  /// 釋放模型資源
  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }
}
