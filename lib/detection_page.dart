import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

/// 物件檢測結果類別
class Detection {
  final List<double> boundingBox; // [left, top, right, bottom]
  final double confidence;
  final int classId;
  final String className;

  Detection({
    required this.boundingBox,
    required this.confidence,
    required this.classId,
    this.className = '',
  });
}

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  CameraController? _cameraController;
  List<CameraDescription>? cameras;
  OrtSession? _session;
  bool _isDetecting = false;
  bool _isProcessing = false;
  List<Detection> _detections = [];
  List<String> _capturedImages = [];

  // 相機和模型尺寸相關參數
  Size? _originalSize;
  final int _modelInputSize = 320; // YOLO11n的輸入尺寸

  // 記憶體優化：重用輸入緩衝區
  late final Float32List _inputBuffer = Float32List(
    1 * 3 * _modelInputSize * _modelInputSize,
  );

  // 效能優化：節流處理器
  final FrameThrottler _throttler = FrameThrottler(
    minInterval: Duration(milliseconds: 100),
  );

  // 偵測參數調整
  final double _confidenceThreshold = 0.3; // 信心度閾值，0.3是推薦值
  final double _iouThreshold = 0.45; // NMS閾值，0.45是推薦值

  // 日誌相關
  SendPort? _logSendPort;
  final List<String> _logBuffer = [];

  // COCO 資料集類別（80個類別）
  final List<String> _cocoClasses = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
  ];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModel();
    _setupLoggingIsolate();
  }

  void _setupLoggingIsolate() {
    final receivePort = ReceivePort();
    Isolate.spawn(_logIsolate, receivePort.sendPort);
    receivePort.listen((message) {
      if (message is SendPort) {
        _logSendPort = message;
        for (var log in _logBuffer) {
          _logSendPort!.send(log);
        }
        _logBuffer.clear();
      }
    });
  }

  static void _logIsolate(SendPort sendPort) {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);
    receivePort.listen((message) {
      print(message);
    });
  }

  void _log(String message) {
    if (_logSendPort != null) {
      _logSendPort!.send(message);
    } else {
      _logBuffer.add(message);
    }
  }

  Future<void> _initCamera() async {
    try {
      cameras = await availableCameras();
      if (cameras == null || cameras!.isEmpty) {
        _log('未找到可用相機');
        return;
      }
      _cameraController = CameraController(
        cameras![0],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup:
            Platform.isAndroid
                ? ImageFormatGroup.yuv420
                : ImageFormatGroup.bgra8888,
      );
      await _cameraController!.initialize();
      await _cameraController!.setFocusMode(FocusMode.auto);

      // 儲存原始預覽尺寸
      _originalSize = Size(
        _cameraController!.value.previewSize!.width,
        _cameraController!.value.previewSize!.height,
      );

      _log('相機初始化完成: 原始尺寸=${_originalSize!.width}x${_originalSize!.height}');

      if (mounted) setState(() {});
    } catch (e) {
      _log('相機初始化失敗: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      // 初始化 ORT 環境
      OrtEnv.instance.init();

      // 建立會話選項
      final sessionOptions = OrtSessionOptions();

      // 設置線程數和性能選項
      sessionOptions.setIntraOpNumThreads(4);
      sessionOptions.setInterOpNumThreads(2);

      // 從資產載入模型
      const assetFileName = 'assets/models/yolo11n320nms.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();

      // 創建 ORT 會話
      _session = OrtSession.fromBuffer(bytes, sessionOptions);
      _log('模型載入成功: YOLOv11n 320x320');
    } catch (e) {
      _log('模型載入失敗: $e');
    }
  }

  Future<void> _startDetection() async {
    if (_isDetecting) {
      await _stopDetection();
      return;
    }
    try {
      setState(() {
        _isDetecting = true;
        _detections.clear();
      });
      await _cameraController!.startImageStream(_processImage);
    } catch (e) {
      if (e is CameraException && e.code == 'CameraAccessDenied') {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('需要相機權限')));
      } else {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('啟動辨識失敗: $e')));
      }
      setState(() {
        _isDetecting = false;
      });
    }
  }

  Future<void> _stopDetection() async {
    setState(() {
      _isDetecting = false;
      _detections.clear();
    });
    await _cameraController!.stopImageStream();
  }

  Future<void> _processImage(CameraImage image) async {
    // 使用節流器控制處理頻率
    if (!_throttler.shouldProcessFrame() || _isProcessing) return;

    _isProcessing = true;
    try {
      // 準備輸入張量
      await _prepareInput(image);

      // 執行推理
      final outputs = await _runInference();

      // 後處理檢測結果
      final detections = _processOutput(outputs);

      if (mounted) {
        setState(() {
          _detections = detections;
        });
      }
    } catch (e) {
      _log('影像處理失敗: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<void> _prepareInput(CameraImage image) async {
    try {
      // 根據圖像格式選擇不同的轉換方法
      img.Image rgbImage;
      if (image.format.group == ImageFormatGroup.yuv420) {
        rgbImage = _convertYUV420ToRGB(image);
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        rgbImage = _convertBGRA8888ToRGB(image);
      } else {
        _log('不支援的圖像格式: ${image.format.group}');
        return;
      }

      // 使用 letterbox 調整圖像尺寸到 320x320，保持寬高比
      final letterboxedImage = _letterboxImage(
        rgbImage,
        _modelInputSize,
        _modelInputSize,
      );

      // 填充輸入緩衝區 (NCHW 格式) [batch, channels, height, width]
      _fillInputBuffer(letterboxedImage);
    } catch (e) {
      _log('準備輸入失敗: $e');
    }
  }

  img.Image _convertYUV420ToRGB(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    // 正確使用步長參數
    final yRowStride = yPlane.bytesPerRow;
    final yPixelStride = yPlane.bytesPerPixel ?? 1;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final output = img.Image(width: width, height: height);

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        final int uvh = (h / 2).floor();
        final int uvw = (w / 2).floor();

        final int yIndex = (h * yRowStride) + (w * yPixelStride);
        final int uIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);
        final int vIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);

        // 確保索引在有效範圍內
        if (yIndex >= yPlane.bytes.length ||
            uIndex >= uPlane.bytes.length ||
            vIndex >= vPlane.bytes.length) {
          continue;
        }

        // YUV 到 RGB 的轉換公式
        final int y = yPlane.bytes[yIndex] & 0xFF;
        final int u = uPlane.bytes[uIndex] & 0xFF;
        final int v = vPlane.bytes[vIndex] & 0xFF;

        // 標準色彩轉換公式
        int r = (y + 1.402 * (v - 128)).round().clamp(0, 255);
        int g = (y - 0.344 * (u - 128) - 0.714 * (v - 128)).round().clamp(
          0,
          255,
        );
        int b = (y + 1.772 * (u - 128)).round().clamp(0, 255);

        output.setPixelRgb(w, h, r, g, b);
      }
    }

    return output;
  }

  img.Image _convertBGRA8888ToRGB(CameraImage image) {
    final bytes = image.planes[0].bytes;

    // 直接從 BGRA 位元組建立圖像
    final originalImage = img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: bytes.buffer,
      order: img.ChannelOrder.bgra,
    );

    // 將 BGRA 轉換為 RGB
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

  img.Image _letterboxImage(
    img.Image image,
    int targetWidth,
    int targetHeight,
  ) {
    // 計算保持寬高比的新尺寸
    final originalRatio = image.width / image.height;
    final targetRatio = targetWidth / targetHeight;

    int newWidth, newHeight;
    if (originalRatio > targetRatio) {
      // 圖像較寬，依寬度縮放
      newWidth = targetWidth;
      newHeight = (targetWidth / originalRatio).round();
    } else {
      // 圖像較高，依高度縮放
      newHeight = targetHeight;
      newWidth = (targetHeight * originalRatio).round();
    }

    // 縮放圖像
    final resizedImage = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );

    // 創建 letterbox 圖像（中心填充）
    final letterboxedImage = img.Image(
      width: targetWidth,
      height: targetHeight,
    );
    // 填充灰色背景
    img.fill(letterboxedImage, color: img.ColorRgb8(114, 114, 114));

    // 計算偏移量以使圖像居中
    final offsetX = (targetWidth - newWidth) ~/ 2;
    final offsetY = (targetHeight - newHeight) ~/ 2;

    // 將調整大小的圖像貼到背景中
    img.compositeImage(
      letterboxedImage,
      resizedImage,
      dstX: offsetX,
      dstY: offsetY,
    );

    return letterboxedImage;
  }

  void _fillInputBuffer(img.Image image) {
    // 將 RGB 圖像填充到 NCHW 格式的輸入緩衝區
    // [batch, channels, height, width]
    int pixelIndex = 0;

    // 每個通道分開存儲
    final int channelSize = _modelInputSize * _modelInputSize;

    // RGB 歸一化
    for (int y = 0; y < _modelInputSize; y++) {
      for (int x = 0; x < _modelInputSize; x++) {
        final pixel = image.getPixel(x, y);

        // 紅色通道 (R) - 正規化為 [0,1]
        _inputBuffer[pixelIndex] = (pixel.r.toDouble() / 255.0);
        // 綠色通道 (G)
        _inputBuffer[pixelIndex + channelSize] = (pixel.g.toDouble() / 255.0);
        // 藍色通道 (B)
        _inputBuffer[pixelIndex + 2 * channelSize] =
            (pixel.b.toDouble() / 255.0);

        pixelIndex++;
      }
    }
  }

  Future<List<OrtValue?>> _runInference() async {
    try {
      // 創建輸入張量 (NCHW 格式)
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        _inputBuffer,
        [1, 3, _modelInputSize, _modelInputSize],
      );

      // 建立推理選項
      final runOptions = OrtRunOptions();

      // 執行推理（輸入名稱為 'images'）
      final inputs = {'images': inputTensor};
      final outputs = await _session?.runAsync(runOptions, inputs) ?? [];

      // 釋放資源
      inputTensor.release();
      runOptions.release();

      return outputs;
    } catch (e) {
      _log('推理執行失敗: $e');
      return [];
    }
  }

  // 修正的輸出處理函數，解決信心度和邊界框問題
  List<Detection> _processOutput(List<OrtValue?> outputs) {
    try {
      if (outputs.isEmpty || outputs[0] == null) {
        _log('無有效輸出數據');
        return [];
      }

      final outputTensor = outputs[0] as OrtValueTensor;
      final data = outputTensor.value as List<dynamic>;

      if (data.isEmpty || data[0].isEmpty) {
        _log('輸出數據為空');
        return [];
      }

      // 檢查輸出形狀
      final dataShape = _getListShape(data);
      _log('原始輸出形狀: $dataShape');

      // 存儲檢測結果
      final List<Detection> detections = [];

      // 根據 YOLO11n 的輸出格式處理 [1, 84, 8400]
      if (dataShape.length >= 2 && dataShape[0] == 1) {
        final batch0 = data[0];

        // 確認輸出格式
        if (batch0 is List) {
          int numClasses = 80; // COCO 資料集的類別數
          int numBBoxValues = 4; // 邊界框值的數量 (cx, cy, w, h)

          // 獲取檢測候選框數量，應該是 8400 (但有些模型可能是 2100 或其他值)
          int numDetections = batch0[0].length;
          _log('檢測候選框數量: $numDetections');

          // 建立候選列表
          List<Map<String, dynamic>> candidates = [];

          // 處理每個檢測框
          for (int i = 0; i < numDetections; i++) {
            try {
              // 提取類別分數 (索引 4 到 numBBoxValues+numClasses-1)
              List<double> classScores = [];
              for (int j = 0; j < numClasses; j++) {
                int scoreIndex = numBBoxValues + j;
                if (scoreIndex < batch0.length) {
                  double score = (batch0[scoreIndex][i] as num).toDouble();
                  // 應用 sigmoid 激活函數 - 這是關鍵修復
                  // score = _sigmoid(score);
                  classScores.add(score);
                }
              }

              // 找到得分最高的類別
              int maxClassId = 0;
              double maxScore = 0.0;
              for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxScore) {
                  maxScore = classScores[j];
                  maxClassId = j;
                }
              }

              // 信心度閾值過濾
              if (maxScore >= _confidenceThreshold) {
                // 從前 4 個通道獲取邊界框坐標 (cx, cy, w, h)
                // 這些是標準化的坐標 [0-1]
                final cx = (batch0[0][i] as num).toDouble();
                final cy = (batch0[1][i] as num).toDouble();
                final w = (batch0[2][i] as num).toDouble();
                final h = (batch0[3][i] as num).toDouble();

                if (i < 5) {
                  // 輸出前5個做調試
                  _log(
                    '座標: cx=$cx, cy=$cy, w=$w, h=$h, 最高信心度: $maxScore (類別=$maxClassId)',
                  );
                }

                // 轉換為左上右下格式 (XYXY) - 這個修復是關鍵
                final left = (cx - w / 2) * _originalSize!.width;
                final top = (cy - h / 2) * _originalSize!.height;
                final right = (cx + w / 2) * _originalSize!.width;
                final bottom = (cy + h / 2) * _originalSize!.height;

                // 裁剪坐標到圖像範圍內
                final clippedLeft = left.clamp(0.0, _originalSize!.width);
                final clippedTop = top.clamp(0.0, _originalSize!.height);
                final clippedRight = right.clamp(0.0, _originalSize!.width);
                final clippedBottom = bottom.clamp(0.0, _originalSize!.height);

                // 添加到候選列表
                candidates.add({
                  'box': [clippedLeft, clippedTop, clippedRight, clippedBottom],
                  'score': maxScore,
                  'class_id': maxClassId,
                });
              }
            } catch (e) {
              _log('處理檢測框 $i 出錯: $e');
            }
          }

          _log('候選檢測數量: ${candidates.length}');

          // 按置信度排序候選檢測
          candidates.sort((a, b) => b['score'].compareTo(a['score']));

          // 建立檢測對象
          for (var candidate in candidates) {
            detections.add(
              Detection(
                boundingBox: List<double>.from(candidate['box']),
                confidence: candidate['score'],
                classId: candidate['class_id'],
                className: _cocoClasses[candidate['class_id']],
              ),
            );
          }

          // 應用非最大抑制 (NMS)
          return _applyNMS(detections);
        } else {
          _log('未支援的輸出格式');
        }
      } else {
        _log('未支援的輸出形狀: ${dataShape.join("x")}');
      }

      return detections;
    } catch (e) {
      _log('後處理失敗: $e');
      return [];
    } finally {
      for (var element in outputs) {
        element?.release();
      }
    }
  }

  // Sigmoid 函數 - 將邏輯值轉換為概率
  double _sigmoid(double x) {
    // 處理數值溢出問題
    if (x > 15) return 1.0; // 避免 exp(-x) 為 0
    if (x < -15) return 0.0; // 避免 exp(-x) 太大
    return 1.0 / (1.0 + exp(-x));
  }

  // 獲取嵌套列表的形狀
  List<int> _getListShape(dynamic list) {
    final shape = <int>[];
    dynamic current = list;

    while (current is List && current.isNotEmpty) {
      shape.add(current.length);
      current = current[0];
    }

    return shape;
  }

  // 非最大抑制 (NMS) - 修正缺少參數問題
  List<Detection> _applyNMS(List<Detection> boxes) {
    if (boxes.isEmpty) return [];

    // 按置信度排序
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));

    final List<Detection> selected = [];
    final Set<int> suppressed = {};

    for (int i = 0; i < boxes.length; i++) {
      if (suppressed.contains(i)) continue;

      selected.add(boxes[i]);

      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed.contains(j)) continue;

        // 只抑制相同類別的框
        if (boxes[i].classId != boxes[j].classId) continue;

        // 計算 IoU
        final double iou = _calculateIOU(
          boxes[i].boundingBox,
          boxes[j].boundingBox,
        );

        if (iou > _iouThreshold) {
          suppressed.add(j);
        }
      }
    }

    return selected;
  }

  double _calculateIOU(List<double> boxA, List<double> boxB) {
    // boxA and boxB format: [left, top, right, bottom]
    final x1A = boxA[0];
    final y1A = boxA[1];
    final x2A = boxA[2];
    final y2A = boxA[3];

    final x1B = boxB[0];
    final y1B = boxB[1];
    final x2B = boxB[2];
    final y2B = boxB[3];

    // 計算交集區域
    final xA = max(x1A, x1B);
    final yA = max(y1A, y1B);
    final xB = min(x2A, x2B);
    final yB = min(y2A, y2B);

    // 檢查是否存在交集
    if (xB <= xA || yB <= yA) return 0.0;

    final interArea = (xB - xA) * (yB - yA);
    final boxAArea = (x2A - x1A) * (y2A - y1A);
    final boxBArea = (x2B - x1B) * (y2B - y1B);

    // 計算 IoU
    final unionArea = boxAArea + boxBArea - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0;
  }

  Future<void> _takePhoto() async {
    try {
      final XFile file = await _cameraController!.takePicture();
      final directory = await getApplicationDocumentsDirectory();
      final filePath = path.join(directory.path, '${DateTime.now()}.jpg');
      await file.saveTo(filePath);
      setState(() {
        _capturedImages.add(filePath);
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('照片已儲存至 $filePath')));
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('拍照失敗: $e')));
    }
  }

  void _showCapturedImages() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ImagePreviewPage(images: _capturedImages),
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _session?.release();
    OrtEnv.instance.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('物件辨識')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('物件辨識')),
      body: Stack(
        children: [
          CameraPreview(_cameraController!),
          if (_isDetecting)
            CustomPaint(
              painter: DetectionPainter(
                _detections,
                _cameraController!.value.previewSize!,
                MediaQuery.of(context).size,
              ),
              child: Container(),
            ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: _startDetection,
                    child: Text(_isDetecting ? '停止辨識' : '開始辨識'),
                  ),
                  ElevatedButton(
                    onPressed: _takePhoto,
                    child: const Text('拍照'),
                  ),
                  ElevatedButton(
                    onPressed: _showCapturedImages,
                    child: const Text('預覽照片'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// 幀節流器 - 限制處理頻率
class FrameThrottler {
  final Duration minInterval;
  DateTime _lastProcessedTime = DateTime.now();

  FrameThrottler({this.minInterval = const Duration(milliseconds: 100)});

  bool shouldProcessFrame() {
    final now = DateTime.now();
    if (now.difference(_lastProcessedTime) >= minInterval) {
      _lastProcessedTime = now;
      return true;
    }
    return false;
  }
}

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size previewSize;
  final Size screenSize;

  // 類別顏色映射
  final List<Color> _colors = [
    Colors.red,
    Colors.green,
    Colors.blue,
    Colors.yellow,
    Colors.orange,
    Colors.purple,
    Colors.teal,
    Colors.pink,
    Colors.indigo,
    Colors.lime,
  ];

  DetectionPainter(this.detections, this.previewSize, this.screenSize);

  @override
  void paint(Canvas canvas, Size size) {
    // 輸出偵錯信息
    print('畫面尺寸: $size, 預覽尺寸: $previewSize, 偵測數量: ${detections.length}');

    if (detections.isEmpty) return;

    // 計算相機預覽到螢幕的縮放
    final double scaleX = size.width / previewSize.width;
    final double scaleY = size.height / previewSize.height;
    final double scale = min(scaleX, scaleY);

    // 計算填充
    final double offsetX = (size.width - previewSize.width * scale) / 2;
    final double offsetY = (size.height - previewSize.height * scale) / 2;

    // 輸出計算結果
    print('縮放: $scale, 填充: X=$offsetX, Y=$offsetY');

    for (var detection in detections) {
      try {
        // 獲取檢測框坐標
        final boundingBox = detection.boundingBox;

        // 將原始坐標映射到螢幕上
        final screenLeft = boundingBox[0] * scale + offsetX;
        final screenTop = boundingBox[1] * scale + offsetY;
        final screenRight = boundingBox[2] * scale + offsetX;
        final screenBottom = boundingBox[3] * scale + offsetY;

        // 忽略無效框
        if (screenRight <= screenLeft || screenBottom <= screenTop) continue;

        // 選擇顏色
        final color = _colors[detection.classId % _colors.length];

        // 繪製檢測框
        final boxPaint =
            Paint()
              ..color = color
              ..style = PaintingStyle.stroke
              ..strokeWidth = 4.0; // 增加線寬，更容易看見

        final rect = Rect.fromLTRB(
          screenLeft,
          screenTop,
          screenRight,
          screenBottom,
        );

        canvas.drawRect(rect, boxPaint);

        // 繪製類別名稱和置信度
        final labelText =
            '${detection.className}: ${(detection.confidence * 100).toStringAsFixed(0)}%';

        // 繪製標籤背景
        final backgroundPaint =
            Paint()
              ..color = color.withOpacity(0.7)
              ..style = PaintingStyle.fill;

        // 繪製標籤文字
        final textStyle = TextStyle(
          color: Colors.white,
          fontSize: 16, // 增加字體大小，更容易閱讀
          fontWeight: FontWeight.bold,
        );
        final textSpan = TextSpan(text: labelText, style: textStyle);
        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();

        // 標籤背景矩形 - 置於框的頂部
        final labelRect = Rect.fromLTWH(
          screenLeft,
          max(0, screenTop - textPainter.height - 4),
          textPainter.width + 8,
          textPainter.height + 4,
        );

        canvas.drawRect(labelRect, backgroundPaint);
        textPainter.paint(
          canvas,
          Offset(labelRect.left + 4, labelRect.top + 2),
        );
      } catch (e) {
        print('繪製偵測框錯誤: $e');
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class ImagePreviewPage extends StatelessWidget {
  final List<String> images;

  const ImagePreviewPage({super.key, required this.images});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('預覽照片')),
      body: GridView.builder(
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3,
        ),
        itemCount: images.length,
        itemBuilder: (context, index) {
          return Image.file(File(images[index]));
        },
      ),
    );
  }
}
