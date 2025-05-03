import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

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
  List<dynamic> _detections = [];
  DateTime _lastProcessTime = DateTime.now();
  List<String> _capturedImages = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModel();
  }

  Future<void> _initCamera() async {
    try {
      cameras = await availableCameras();
      if (cameras == null || cameras!.isEmpty) {
        print('未找到可用相機');
        return;
      }
      _cameraController = CameraController(
        cameras![0],
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _cameraController!.initialize();
      await _cameraController!.setFocusMode(FocusMode.auto);
      if (mounted) setState(() {});
    } catch (e) {
      print('相機初始化失敗: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      OrtEnv.instance.init();
      final sessionOptions = OrtSessionOptions();
      const assetFileName = 'assets/models/yolo11n320.onnx';
      final rawAssetFile = await DefaultAssetBundle.of(
        context,
      ).load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);
      print('模型載入成功');
    } catch (e) {
      print('模型載入失敗: $e');
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
    if (_isProcessing) return;
    _isProcessing = true;
    try {
      print(
        '處理圖像格式: ${image.format.group}, 寬: ${image.width}, 高: ${image.height}',
      );
      final input = await _prepareInput(image);
      if (input.isEmpty) {
        print('輸入數據為空，檢查圖像解碼');
        return;
      }
      final outputs = await _runInference(input);
      final detections = _postProcess(outputs);
      if (mounted) {
        setState(() {
          _detections = detections;
        });
      }
    } catch (e) {
      print('影像處理失敗: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<Float32List> _prepareInput(CameraImage image) async {
    img.Image? imgImage;
    try {
      print('圖像格式: ${image.format.group}, 平面數量: ${image.planes.length}');
      if (image.format.group == ImageFormatGroup.bgra8888) {
        final bytes = image.planes[0].bytes;
        print(
          'BGRA8888 bytes length: ${bytes.length}, expected: ${image.width * image.height * 4}',
        );
        if (bytes.length != image.width * image.height * 4) {
          print('Bytes length mismatch');
          return Float32List(0);
        }
        imgImage = img.Image.fromBytes(
          width: image.width,
          height: image.height,
          bytes: bytes.buffer,
          order: img.ChannelOrder.bgra,
        );
      } else if (image.format.group == ImageFormatGroup.yuv420) {
        imgImage = _convertYUV420toImage(image);
      } else {
        print('不支援的圖像格式: ${image.format.group}');
        return Float32List(0);
      }

      if (imgImage == null) {
        print('圖像轉換失敗: imgImage 為 null');
        return Float32List(0);
      }

      imgImage = img.copyResize(imgImage, width: 320, height: 320);
      final input = Float32List(1 * 3 * 320 * 320);
      int pixelIndex = 0;
      for (int y = 0; y < 320; y++) {
        for (int x = 0; x < 320; x++) {
          final pixel = imgImage.getPixel(x, y);
          final red = pixel.r.toInt();
          final green = pixel.g.toInt();
          final blue = pixel.b.toInt();
          input[pixelIndex] = red / 255.0;
          input[pixelIndex + 1] = green / 255.0;
          input[pixelIndex + 2] = blue / 255.0;
          pixelIndex += 3;
        }
      }
      print('輸入數據大小: ${input.lengthInBytes} 位元組');
      return input;
    } catch (e) {
      print('準備輸入失敗: $e');
      return Float32List(0);
    }
  }

  img.Image _convertYUV420toImage(CameraImage image) {
    const shift = (0xFF << 24);
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final img.Image imgImage = img.Image(width: width, height: height);
    try {
      print('Y 平面長度: ${image.planes[0].bytes.length}, 預期: ${width * height}');
      print(
        'U 平面長度: ${image.planes[1].bytes.length}, 預期: ${(width ~/ 2) * (height ~/ 2)}',
      );
      print(
        'V 平面長度: ${image.planes[2].bytes.length}, 預期: ${(width ~/ 2) * (height ~/ 2)}',
      );
      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
          final int uvIndex =
              uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
          final int index = y * width + x;
          if (index >= image.planes[0].bytes.length) {
            print('Y 平面索引超出範圍: $index >= ${image.planes[0].bytes.length}');
            continue;
          }
          final yp = image.planes[0].bytes[index];
          final up =
              uvIndex < image.planes[1].bytes.length
                  ? image.planes[1].bytes[uvIndex]
                  : 128;
          final vp =
              uvIndex < image.planes[2].bytes.length
                  ? image.planes[2].bytes[uvIndex]
                  : 128;
          int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
              .round()
              .clamp(0, 255);
          int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
          imgImage.setPixelRgba(x, y, r, g, b, shift);
        }
      }
      return imgImage;
    } catch (e) {
      print('YUV420 轉換失敗: $e');
      return img.Image(width: 1, height: 1);
    }
  }

  Future<List<OrtValue?>> _runInference(Float32List input) async {
    try {
      print('當前可用記憶體: ${ProcessInfo.currentRss ~/ 1024} KB');
      final shape = [1, 3, 320, 320];
      final inputOrt = OrtValueTensor.createTensorWithDataList(input, shape);
      final inputs = {'images': inputOrt};
      final runOptions = OrtRunOptions();
      final outputs = await _session?.runAsync(runOptions, inputs) ?? [];
      print('輸出數量: ${outputs.length}');
      for (int i = 0; i < outputs.length; i++) {
        print('輸出 $i 類型: ${outputs[i]?.runtimeType}');
      }
      inputOrt.release();
      runOptions.release();
      return outputs;
    } catch (e) {
      print('推理失敗: $e');
      return [];
    }
  }

  List<Map<String, dynamic>> _postProcess(List<OrtValue?> outputs) {
    try {
      if (outputs.isEmpty || outputs[0] == null) {
        print('無有效輸出數據');
        return [];
      }
      if (outputs[0] is! OrtValueTensor) {
        print('輸出不是張量，類型: ${outputs[0]?.runtimeType}');
        return [];
      }
      final outputTensor = outputs[0] as OrtValueTensor;
      final data = outputTensor.value as List<List<List<double>>>;
      if (data.isEmpty || data[0].isEmpty) {
        print('輸出數據為空');
        return [];
      }
      final detections = <Map<String, dynamic>>[];
      final batchData = data[0]; // 假設 batch_size = 1
      print('檢測數量: ${batchData.length}');
      for (var detection in batchData) {
        if (detection.length < 5) {
          print('檢測屬性不足: ${detection.length} < 5');
          continue;
        }
        final confidence = detection[4];
        if (confidence > 0.5) {
          final box = detection.sublist(0, 4);
          final classProbs = detection.sublist(5);
          if (classProbs.isEmpty) {
            print('類別概率為空');
            continue;
          }
          final maxClassProb = classProbs.reduce((a, b) => a > b ? a : b);
          final classIndex = classProbs.indexOf(maxClassProb);
          detections.add({
            'box': box,
            'confidence': confidence,
            'class': classIndex,
          });
        }
      }
      return detections;
    } catch (e) {
      print('後處理失敗: $e');
      return [];
    } finally {
      for (var element in outputs) {
        element?.release();
      }
    }
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
              painter: DetectionPainter(_detections),
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

class DetectionPainter extends CustomPainter {
  final List<dynamic> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final paint =
        Paint()
          ..color = Colors.red
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0;
    for (var detection in detections) {
      final box = detection['box'] as List<double>;
      final cx = box[0] * size.width / 320;
      final cy = box[1] * size.height / 320;
      final w = box[2] * size.width / 320;
      final h = box[3] * size.height / 320;
      final rect = Rect.fromLTWH(cx - w / 2, cy - h / 2, w, h);
      canvas.drawRect(rect, paint);
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
