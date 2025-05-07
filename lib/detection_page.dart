import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
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
  double? _originalWidth;
  double? _originalHeight;
  final int _modelInputSize = 320; // YOLO11n的輸入尺寸
  double? _scalingFactor; // 縮放因子
  double? _offsetX; // letterbox 水平偏移量
  double? _offsetY; // letterbox 垂直偏移量

  // 記憶體優化：重用輸入緩衝區
  late final Float32List _inputBuffer = Float32List(
    1 * 3 * _modelInputSize * _modelInputSize,
  );

  // 效能優化：節流處理器
  final FrameThrottler _throttler = FrameThrottler(
    minInterval: Duration(milliseconds: 100),
  );

  // 偵測參數調整
  final double _confidenceThreshold = 0.3; // 降低閾值以獲得更多檢測
  final double _iouThreshold = 0.45; // NMS閾值，0.45是推薦值

  // 日誌相關
  SendPort? _logSendPort;
  final List<String> _logBuffer = [];

  // FPS 計數
  int _frameCount = 0;
  int _fps = 0;

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
    // FPS 計數初始化
    Timer.periodic(Duration(seconds: 1), (timer) {
      setState(() {
        _fps = _frameCount;
        _frameCount = 0;
      });
    });
    void frameCallback(Duration timeStamp) {
      _frameCount++;
      SchedulerBinding.instance.scheduleFrameCallback(frameCallback);
    }

    SchedulerBinding.instance.scheduleFrameCallback(frameCallback);
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
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup:
            Platform.isAndroid
                ? ImageFormatGroup.yuv420
                : ImageFormatGroup.bgra8888,
      );
      await _cameraController!.initialize();
      await _cameraController!.setFocusMode(FocusMode.auto);
      _log('相機初始化完成');
      if (mounted) setState(() {});
    } catch (e) {
      _log('相機初始化失敗: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      OrtEnv.instance.init();
      const assetFileName = 'assets/models/yolo11n320nms.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(
        bytes,
        OrtSessionOptions()
          ..setIntraOpNumThreads(4)
          ..setInterOpNumThreads(2),
      );
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
    if (!_throttler.shouldProcessFrame() || _isProcessing) return;
    _isProcessing = true;
    try {
      // 設置原始圖像尺寸
      _originalWidth = image.width.toDouble();
      _originalHeight = image.height.toDouble();

      // 計算 letterboxing 參數
      final targetWidth = _modelInputSize.toDouble();
      final targetHeight = _modelInputSize.toDouble();
      double newWidth, newHeight;
      if (_originalWidth! / _originalHeight! > 1) {
        newWidth = targetWidth;
        newHeight = targetWidth * _originalHeight! / _originalWidth!;
      } else {
        newHeight = targetHeight;
        newWidth = targetHeight * _originalWidth! / _originalHeight!;
      }
      _scalingFactor = min(
        targetWidth / _originalWidth!,
        targetHeight / _originalHeight!,
      );
      _offsetX = (targetWidth - newWidth) / 2;
      _offsetY = (targetHeight - newHeight) / 2;

      _log(
        '處理圖像: 原始尺寸=${_originalWidth}x${_originalHeight}, '
        '縮放因子=$_scalingFactor, 偏移=($_offsetX, $_offsetY)',
      );

      // 準備輸入
      await _prepareInput(image);

      // 執行推理
      final outputs = await _runInference();

      // 處理輸出
      final detections = _processOutput(outputs);

      // 轉換邊界框到預覽坐標（假設 90 度順時針旋轉）
      final transformedDetections =
          detections.map((det) {
            final left = det.boundingBox[1]; // top -> left
            final top =
                _originalWidth! - det.boundingBox[2]; // W - right -> top
            final right = det.boundingBox[3]; // bottom -> right
            final bottom =
                _originalWidth! - det.boundingBox[0]; // W - left -> bottom
            return Detection(
              boundingBox: [left, top, right, bottom],
              confidence: det.confidence,
              classId: det.classId,
              className: det.className,
            );
          }).toList();

      // 記錄前 3 名檢測
      if (transformedDetections.isNotEmpty) {
        transformedDetections.sort(
          (a, b) => b.confidence.compareTo(a.confidence),
        );
        final top3 = transformedDetections.take(3).toList();
        for (int i = 0; i < top3.length; i++) {
          var det = top3[i];
          _log(
            '前 ${i + 1} 名檢測: 類別=${det.className}, '
            '信心分數=${(det.confidence * 100).toStringAsFixed(2)}%, '
            '邊界框=${det.boundingBox}',
          );
        }
      }

      if (mounted) {
        setState(() {
          _detections = transformedDetections;
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
      _log('CameraImage 尺寸: ${image.width}x${image.height}');
      img.Image rgbImage;
      if (image.format.group == ImageFormatGroup.yuv420) {
        rgbImage = _convertYUV420ToRGB(image);
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        rgbImage = _convertBGRA8888ToRGB(image);
      } else {
        _log('不支援的圖像格式: ${image.format.group}');
        return;
      }
      final letterboxedImage = _letterboxImage(
        rgbImage,
        _modelInputSize,
        _modelInputSize,
      );
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
        if (yIndex >= yPlane.bytes.length ||
            uIndex >= uPlane.bytes.length ||
            vIndex >= vPlane.bytes.length) {
          continue;
        }
        final int y = yPlane.bytes[yIndex] & 0xFF;
        final int u = uPlane.bytes[uIndex] & 0xFF;
        final int v = vPlane.bytes[vIndex] & 0xFF;
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

  img.Image _letterboxImage(
    img.Image image,
    int targetWidth,
    int targetHeight,
  ) {
    final originalRatio = image.width / image.height;
    final targetRatio = targetWidth / targetHeight;
    int newWidth, newHeight;
    if (originalRatio > targetRatio) {
      newWidth = targetWidth;
      newHeight = (targetWidth / originalRatio).round();
    } else {
      newHeight = targetHeight;
      newWidth = (targetHeight * originalRatio).round();
    }
    final resizedImage = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );
    final letterboxedImage = img.Image(
      width: targetWidth,
      height: targetHeight,
    );
    img.fill(letterboxedImage, color: img.ColorRgb8(114, 114, 114));
    final offsetX = (targetWidth - newWidth) ~/ 2;
    final offsetY = (targetHeight - newHeight) ~/ 2;
    img.compositeImage(
      letterboxedImage,
      resizedImage,
      dstX: offsetX,
      dstY: offsetY,
    );
    return letterboxedImage;
  }

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
      _log('推理執行失敗: $e');
      return [];
    }
  }

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
      final dataShape = _getListShape(data);
      _log('原始輸出形狀: $dataShape');
      final List<Detection> detections = [];
      if (dataShape.length >= 2 && dataShape[0] == 1) {
        final batch0 = data[0];
        if (batch0 is List) {
          int numClasses = 80;
          int numBBoxValues = 4;
          int numDetections = batch0[0].length;
          _log('檢測候選框數量: $numDetections');
          List<Map<String, dynamic>> candidates = [];
          for (int i = 0; i < numDetections; i++) {
            try {
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
              if (maxScore >= _confidenceThreshold) {
                final cx = (batch0[0][i] as num).toDouble();
                final cy = (batch0[1][i] as num).toDouble();
                final w = (batch0[2][i] as num).toDouble();
                final h = (batch0[3][i] as num).toDouble();
                final cx_pixel = cx * _modelInputSize.toDouble();
                final cy_pixel = cy * _modelInputSize.toDouble();
                final cx_original = (cx_pixel - _offsetX!) / _scalingFactor!;
                final cy_original = (cy_pixel - _offsetY!) / _scalingFactor!;
                final w_original =
                    (w * _modelInputSize.toDouble()) / _scalingFactor!;
                final h_original =
                    (h * _modelInputSize.toDouble()) / _scalingFactor!;
                final left = cx_original - w_original / 2;
                final top = cy_original - h_original / 2;
                final right = cx_original + w_original / 2;
                final bottom = cy_original + h_original / 2;
                final clippedLeft = left.clamp(0.0, _originalWidth!);
                final clippedTop = top.clamp(0.0, _originalHeight!);
                final clippedRight = right.clamp(0.0, _originalWidth!);
                final clippedBottom = bottom.clamp(0.0, _originalHeight!);
                if (i < 5) {
                  _log(
                    '原始坐標: cx=$cx, cy=$cy, w=$w, h=$h, '
                    '映射後: cx=$cx_original, cy=$cy_original, w=$w_original, h=$h_original, '
                    '最高信心度: $maxScore (類別=$maxClassId)',
                  );
                }
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
          candidates.sort((a, b) => b['score'].compareTo(a['score']));
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
          _log('NMS 前檢測數量: ${detections.length}');
          final nmsDetections = _applyNMS(detections);
          _log('NMS 後檢測數量: ${nmsDetections.length}');
          return nmsDetections;
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

  List<int> _getListShape(dynamic list) {
    final shape = <int>[];
    dynamic current = list;
    while (current is List && current.isNotEmpty) {
      shape.add(current.length);
      current = current[0];
    }
    return shape;
  }

  List<Detection> _applyNMS(List<Detection> boxes) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<Detection> selected = [];
    final Set<int> suppressed = {};
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
        if (iou > _iouThreshold) {
          suppressed.add(j);
        }
      }
    }
    return selected;
  }

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
                Size(
                  _originalHeight ?? 640.0,
                  _originalWidth ?? 480.0,
                ), // 預覽坐標使用旋轉後尺寸
                MediaQuery.of(context).size,
              ),
              child: Container(),
            ),
          Positioned(
            top: 10,
            left: 10,
            child: Text(
              'FPS: $_fps',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
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
    print('畫面尺寸: $size, 預覽尺寸: $previewSize, 偵測數量: ${detections.length}');
    if (detections.isEmpty) return;
    final double scaleX = size.width / previewSize.width;
    final double scaleY = size.height / previewSize.height;
    final double scale = min(scaleX, scaleY);
    final double offsetX = (size.width - previewSize.width * scale) / 2;
    final double offsetY = (size.height - previewSize.height * scale) / 2;
    print('縮放: $scale, 填充: X=$offsetX, Y=$offsetY');

    for (var detection in detections) {
      try {
        final boundingBox = detection.boundingBox;
        final screenLeft = boundingBox[0] * scale + offsetX;
        final screenTop = boundingBox[1] * scale + offsetY;
        final screenRight = boundingBox[2] * scale + offsetX;
        final screenBottom = boundingBox[3] * scale + offsetY;
        if (screenRight <= screenLeft || screenBottom <= screenTop) continue;
        final color = _colors[detection.classId % _colors.length];
        final boxPaint =
            Paint()
              ..color = color
              ..style = PaintingStyle.stroke
              ..strokeWidth = 4.0;
        final rect = Rect.fromLTRB(
          screenLeft,
          screenTop,
          screenRight,
          screenBottom,
        );
        canvas.drawRect(rect, boxPaint);
        print(
          '檢測到物件: ${detection.className}, 信心度: ${(detection.confidence * 100).toStringAsFixed(2)}%',
        );
        final labelText =
            '${detection.className}: ${(detection.confidence * 100).toStringAsFixed(0)}%';
        final backgroundPaint =
            Paint()
              ..color = color.withOpacity(0.7)
              ..style = PaintingStyle.fill;
        final textStyle = TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        );
        final textSpan = TextSpan(text: labelText, style: textStyle);
        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
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
