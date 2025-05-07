import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/scheduler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'dart:io';
import 'fps_handler.dart';
import 'logging.dart';
import 'onnx_detector.dart';
import 'detection_painter.dart';
import 'models.dart';

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  CameraController? _cameraController;
  List<CameraDescription>? cameras;
  bool _isDetecting = false;
  bool _isProcessing = false;
  List<Detection> _detections = [];
  List<String> _capturedImages = [];
  double? _imageWidth;
  double? _imageHeight;

  final FrameThrottler _throttler = FrameThrottler(
    minInterval: Duration(milliseconds: 100),
  );
  FPSHandler _fpsHandler = FPSHandler();
  Logger _logger = Logger();
  OnnxDetector _onnxDetector = OnnxDetector();
  int _fps = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _onnxDetector.loadModel();
    _logger.setup();
    _fpsHandler.start((fps) {
      setState(() {
        _fps = fps;
      });
    });
    SchedulerBinding.instance.scheduleFrameCallback(_frameCallback);
    _logger.log('物件辨識應用程式啟動');
  }

  void _frameCallback(Duration timeStamp) {
    _fpsHandler.incrementFrameCount();
    SchedulerBinding.instance.scheduleFrameCallback(_frameCallback);
  }

  Future<void> _initCamera() async {
    try {
      cameras = await availableCameras();
      if (cameras == null || cameras!.isEmpty) {
        _logger.log('未找到可用相機');
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
      _logger.log('相機初始化完成');
      if (mounted) setState(() {});
    } catch (e) {
      _logger.log('相機初始化失敗: $e');
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
      _imageWidth = image.width.toDouble();
      _imageHeight = image.height.toDouble();

      // 使用改進過的detector進行檢測
      var detections = await _onnxDetector.detect(image);

      if (detections.isNotEmpty) {
        detections.sort((a, b) => b.confidence.compareTo(a.confidence));
        final top3 = detections.take(3).toList();
        for (int i = 0; i < top3.length; i++) {
          var det = top3[i];
          _logger.log(
            '前 ${i + 1} 名檢測: 類別=${det.className}, '
            '信心分數=${(det.confidence * 100).toStringAsFixed(2)}%, '
            '邊界框=${det.boundingBox}',
          );
        }
      }

      if (mounted) {
        setState(() {
          _detections = detections;
        });
      }
    } catch (e) {
      _logger.log('影像處理失敗: $e');
    } finally {
      _isProcessing = false;
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
    _onnxDetector.dispose();
    _fpsHandler.stop();
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
                // 使用相機實際尺寸，而不是固定值
                Size(
                  _cameraController!.value.previewSize?.height ?? 0,
                  _cameraController!.value.previewSize?.width ?? 0,
                ),
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
