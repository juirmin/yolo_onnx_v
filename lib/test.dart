import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnx/onnx.dart' as onnx; // Hypothetical ONNX package for Dart
import 'package:path_provider/path_provider.dart';
import 'package:collection/collection.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: const HomeScreen());
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  String? _outputPath;

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await _processImage(_image!);
    }
  }

  Future<void> _processImage(File imageFile) async {
    final yolo = YOLO(
      modelPath: 'assets/yolo11n-seg.onnx', // Ensure model is in assets
      confThres: 0.3,
      iouThres: 0.5,
    );
    final image = img.decodeImage(await imageFile.readAsBytes())!;
    final (boxes, scores, classIds, masks) = await yolo.segmentObjects(image);

    // Process output image
    img.Image outputImage = img.copy(image); // Clone original image
    for (int i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final mask = masks[i];

      // Draw rectangle
      outputImage = img.drawRect(
        outputImage,
        x1: box[0].toInt(),
        y1: box[1].toInt(),
        x2: box[2].toInt(),
        y2: box[3].toInt(),
        color: img.ColorRgb8(0, 255, 0),
        thickness: 2,
      );

      // Apply mask
      final maskImage = img.Image(
        width: outputImage.width,
        height: outputImage.height,
        format: img.Format.uint8,
        numChannels: 3,
      );
      for (int y = 0; y < maskImage.height; y++) {
        for (int x = 0; x < maskImage.width; x++) {
          final maskValue = mask[y][x] * 255.0;
          if (maskValue > 0) {
            maskImage.setPixelRgb(x, y, maskValue.toInt(), 0, 0);
          }
        }
      }
      outputImage = img.compositeImage(outputImage, maskImage, dstAlpha: 0.5);
    }

    // Save output
    final directory = await getTemporaryDirectory();
    final outputFile = File('${directory.path}/segtest2.jpg');
    await outputFile.writeAsBytes(img.encodeJpg(outputImage));
    setState(() {
      _outputPath = outputFile.path;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLOv11 Segmentation')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? const Text('No image selected.')
                : Image.file(_image!, height: 200),
            const SizedBox(height: 20),
            _outputPath == null
                ? const Text('No output image.')
                : Image.file(File(_outputPath!), height: 200),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Pick Image'),
            ),
          ],
        ),
      ),
    );
  }
}

// Utility functions
List<int> nms(
  List<List<double>> boxes,
  List<double> scores,
  double iouThreshold,
) {
  final sortedIndices =
      scores.asMap().entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
  final keepBoxes = <int>[];

  while (sortedIndices.isNotEmpty) {
    final boxId = sortedIndices.first.key;
    keepBoxes.add(boxId);

    if (sortedIndices.length == 1) break;

    final ious = computeIou(
      boxes[boxId],
      boxes,
      sortedIndices.skip(1).map((e) => e.key).toList(),
    );
    final keepIndices =
        ious
            .asMap()
            .entries
            .where((e) => e.value < iouThreshold)
            .map((e) => e.key)
            .toList();

    sortedIndices.removeAt(0);
    sortedIndices.retainWhere(
      (e) => keepIndices.contains(sortedIndices.indexOf(e)),
    );
  }

  return keepBoxes;
}

List<double> computeIou(
  List<double> box,
  List<List<double>> boxes,
  List<int> indices,
) {
  final ious = <double>[];
  for (final idx in indices) {
    final otherBox = boxes[idx];

    final xmin = max(box[0], otherBox[0]);
    final ymin = max(box[1], otherBox[1]);
    final xmax = min(box[2], otherBox[2]);
    final ymax = min(box[3], otherBox[3]);

    final intersectionArea = max(0.0, xmax - xmin) * max(0.0, ymax - ymin);
    final boxArea = (box[2] - box[0]) * (box[3] - box[1]);
    final otherBoxArea =
        (otherBox[2] - otherBox[0]) * (otherBox[3] - otherBox[1]);
    final unionArea = boxArea + otherBoxArea - intersectionArea;

    final iou =
        intersectionArea / (unionArea + 1e-10); // Avoid division by zero
    ious.add(iou);
  }
  return ious;
}

List<List<double>> xywh2xyxy(List<List<double>> boxes) {
  return boxes.map((box) {
    return [
      box[0] - box[2] / 2, // x1
      box[1] - box[3] / 2, // y1
      box[0] + box[2] / 2, // x2
      box[1] + box[3] / 2, // y2
    ];
  }).toList();
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

class YOLO {
  final double confThreshold;
  final double iouThreshold;
  final int numMasks;
  late onnx.InferenceSession session;
  late List<String> inputNames;
  late List<String> outputNames;
  late int inputHeight;
  late int inputWidth;
  late int imgHeight;
  late int imgWidth;
  List<List<double>> boxes = [];
  List<double> scores = [];
  List<int> classIds = [];
  List<List<List<double>>> maskMaps = [];

  YOLO({
    required String modelPath,
    this.confThreshold = 0.7,
    this.iouThreshold = 0.5,
    this.numMasks = 32,
  }) {
    initializeModel(modelPath);
  }

  void initializeModel(String modelPath) {
    // Note: Adjust based on actual ONNX package
    session = onnx.InferenceSession.fromAsset(
      modelPath,
      providers: ['CPUExecutionProvider'],
    );
    getInputDetails();
    getOutputDetails();
  }

  Future<
    (List<List<double>>, List<double>, List<int>, List<List<List<double>>>)
  >
  segmentObjects(img.Image image) async {
    final inputTensor = await prepareInput(image);
    final outputs = await inference(inputTensor);
    final (boxes, scores, classIds, maskPred) = processBoxOutput(outputs[0]);
    final maskMaps = processMaskOutput(maskPred, outputs[1]);
    this.boxes = boxes;
    this.scores = scores;
    this.classIds = classIds;
    this.maskMaps = maskMaps;
    return (boxes, scores, classIds, maskMaps);
  }

  Future<List<List<List<double>>>> prepareInput(img.Image image) async {
    imgHeight = image.height;
    imgWidth = image.width;

    // Explicitly convert to RGB
    final rgbImage = img.Image.from(image)
      ..convert(numChannels: 3); // Ensure RGB format

    // Resize image
    final inputImage = img.copyResize(
      rgbImage,
      width: inputWidth,
      height: inputHeight,
      interpolation: img.Interpolation.linear,
    );

    // Prepare input tensor [1, 3, inputHeight, inputWidth]
    final inputTensor = List.generate(
      1,
      (_) => List.generate(
        3,
        (_) => List.generate(inputHeight, (_) => List.filled(inputWidth, 0.0)),
      ),
    );
    for (int y = 0; y < inputHeight; y++) {
      for (int x = 0; x < inputWidth; x++) {
        final pixel = inputImage.getPixel(x, y);
        inputTensor[0][0][y][x] = pixel.r / 255.0; // R
        inputTensor[0][1][y][x] = pixel.g / 255.0; // G
        inputTensor[0][2][y][x] = pixel.b / 255.0; // B
      }
    }

    return inputTensor;
  }

  Future<List<dynamic>> inference(List<List<List<double>>> inputTensor) async {
    final input = {inputNames[0]: inputTensor};
    final outputs = await session.run(input);
    return outputs;
  }

  (List<List<double>>, List<double>, List<int>, List<List<double>>)
  processBoxOutput(dynamic boxOutput) {
    // Assume boxOutput is [1, num_predictions, num_classes + 4 + num_masks]
    final predictions = _transpose(
      _squeeze(boxOutput),
    ); // Shape: [num_predictions, num_classes + 4 + num_masks]
    final numClasses = predictions[0].length - numMasks - 4;

    // Filter by confidence
    final scores =
        predictions
            .map((pred) => pred.sublist(4, 4 + numClasses).reduce(max))
            .toList();
    final filtered = <List<double>>[];
    final filteredScores = <double>[];
    for (int i = 0; i < predictions.length; i++) {
      if (scores[i] > confThreshold) {
        filtered.add(predictions[i]);
        filteredScores.add(scores[i]);
      }
    }

    if (filtered.isEmpty) {
      return ([], [], [], []);
    }

    // Split predictions
    final boxPredictions =
        filtered.map((pred) => pred.sublist(0, 4 + numClasses)).toList();
    final maskPredictions =
        filtered.map((pred) => pred.sublist(4 + numClasses)).toList();

    // Get class IDs
    final classIds =
        boxPredictions
            .map(
              (pred) =>
                  pred
                      .sublist(4)
                      .asMap()
                      .entries
                      .reduce((a, b) => a.value > b.value ? a : b)
                      .key,
            )
            .toList();

    // Extract boxes
    var boxes = extractBoxes(boxPredictions);

    // Apply NMS
    final indices = nms(boxes, filteredScores, iouThreshold);

    return (
      boxes
          .asMap()
          .entries
          .where((e) => indices.contains(e.key))
          .map((e) => e.value)
          .toList(),
      filteredScores
          .asMap()
          .entries
          .where((e) => indices.contains(e.key))
          .map((e) => e.value)
          .toList(),
      classIds
          .asMap()
          .entries
          .where((e) => indices.contains(e.key))
          .map((e) => e.value)
          .toList(),
      maskPredictions
          .asMap()
          .entries
          .where((e) => indices.contains(e.key))
          .map((e) => e.value)
          .toList(),
    );
  }

  List<List<List<double>>> processMaskOutput(
    List<List<double>> maskPredictions,
    dynamic maskOutput,
  ) {
    if (maskPredictions.isEmpty) return [];

    // Assume maskOutput is [1, num_mask, mask_height, mask_width]
    final maskOut = _squeeze(
      maskOutput,
    ); // Shape: [num_mask, mask_height, mask_width]
    final numMask = maskOut.length;
    final maskHeight = maskOut[0].length;
    final maskWidth = maskOut[0][0].length;

    // Compute masks: sigmoid(mask_predictions @ mask_output)
    final masks = _matrixMultiply(maskPredictions, _flattenMaskOutput(maskOut));
    final reshapedMasks = _reshapeMasks(masks, maskHeight, maskWidth);

    // Downscale boxes
    final scaleBoxes = rescaleBoxes(
      boxes,
      (inputHeight, inputWidth),
      (maskHeight, maskWidth),
    );

    // Create mask maps
    final maskMaps = List.generate(
      scaleBoxes.length,
      (_) => List.generate(imgHeight, (_) => List.filled(imgWidth, 0.0)),
    );
    final blurSize = (imgWidth ~/ maskWidth, imgHeight ~/ maskHeight);

    for (int i = 0; i < scaleBoxes.length; i++) {
      final scaleX1 = scaleBoxes[i][0].floor().toInt();
      final scaleY1 = scaleBoxes[i][1].floor().toInt();
      final scaleX2 = scaleBoxes[i][2].ceil().toInt();
      final scaleY2 = scaleBoxes[i][3].ceil().toInt();

      final x1 = boxes[i][0].floor().toInt();
      final y1 = boxes[i][1].floor().toInt();
      final x2 = boxes[i][2].ceil().toInt();
      final y2 = boxes[i][3].ceil().toInt();

      // Crop and resize mask
      final scaleCropMask = _cropMask(
        reshapedMasks[i],
        scaleY1,
        scaleY2,
        scaleX1,
        scaleX2,
      );
      final cropMask = _resizeMask(scaleCropMask, x2 - x1, y2 - y1);

      // Apply blur (approximate with average filter)
      final blurredMask = _blurMask(cropMask, blurSize);

      // Threshold mask
      for (int y = 0; y < y2 - y1; y++) {
        for (int x = 0; x < x2 - x1; x++) {
          if (y + y1 < imgHeight && x + x1 < imgWidth) {
            maskMaps[i][y + y1][x + x1] = blurredMask[y][x] > 0.5 ? 1.0 : 0.0;
          }
        }
      }
    }

    return maskMaps;
  }

  List<List<double>> extractBoxes(List<List<double>> boxPredictions) {
    var boxes = boxPredictions.map((pred) => pred.sublist(0, 4)).toList();
    boxes = rescaleBoxes(
      boxes,
      (inputHeight, inputWidth),
      (imgHeight, imgWidth),
    );
    boxes = xywh2xyxy(boxes);
    boxes =
        boxes
            .map(
              (box) => [
                box[0].clamp(0, imgWidth.toDouble()).toDouble(),
                box[1].clamp(0, imgHeight.toDouble()).toDouble(),
                box[2].clamp(0, imgWidth.toDouble()).toDouble(),
                box[3].clamp(0, imgHeight.toDouble()).toDouble(),
              ],
            )
            .toList();
    return boxes;
  }

  void getInputDetails() {
    final modelInputs = session.getInputs();
    inputNames = modelInputs.map((input) => input.name).toList();
    final inputShape = modelInputs[0].shape;
    inputHeight = inputShape[2];
    inputWidth = inputShape[3];
  }

  void getOutputDetails() {
    final modelOutputs = session.getOutputs();
    outputNames = modelOutputs.map((output) => output.name).toList();
  }

  static List<List<double>> rescaleBoxes(
    List<List<double>> boxes,
    (int, int) inputShape,
    (int, int) imageShape,
  ) {
    final input = [inputShape.$2, inputShape.$1, inputShape.$2, inputShape.$1];
    final scale = [imageShape.$2, imageShape.$1, imageShape.$2, imageShape.$1];
    return boxes.map((box) {
      return List.generate(4, (i) => (box[i] / input[i]) * scale[i]);
    }).toList();
  }

  // Helper functions
  List<List<double>> _squeeze(dynamic array) {
    // Convert to List<List<double>> assuming [1, ...] shape
    return (array as List).first as List<List<double>>;
  }

  List<List<double>> _transpose(List<List<double>> matrix) {
    return List.generate(
      matrix[0].length,
      (i) => matrix.map((row) => row[i]).toList(),
    );
  }

  List<List<double>> _matrixMultiply(
    List<List<double>> a,
    List<List<double>> b,
  ) {
    final result = List.generate(
      a.length,
      (_) => List.filled(b[0].length, 0.0),
    );
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < b[0].length; j++) {
        for (int k = 0; k < a[0].length; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
        result[i][j] = sigmoid(result[i][j]);
      }
    }
    return result;
  }

  List<List<double>> _flattenMaskOutput(List<List<List<double>>> maskOutput) {
    final numMask = maskOutput.length;
    final maskHeight = maskOutput[0].length;
    final maskWidth = maskOutput[0][0].length;
    return List.generate(
      numMask,
      (i) => maskOutput[i].expand((row) => row).toList(),
    );
  }

  List<List<List<double>>> _reshapeMasks(
    List<List<double>> masks,
    int height,
    int width,
  ) {
    return masks.map((mask) {
      final reshaped = List.generate(height, (_) => List.filled(width, 0.0));
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          reshaped[i][j] = mask[i * width + j];
        }
      }
      return reshaped;
    }).toList();
  }

  List<List<double>> _cropMask(
    List<List<double>> mask,
    int y1,
    int y2,
    int x1,
    int x2,
  ) {
    return List.generate(y2 - y1, (i) => mask[y1 + i].sublist(x1, x2));
  }

  List<List<double>> _resizeMask(
    List<List<double>> mask,
    int width,
    int height,
  ) {
    final srcHeight = mask.length;
    final srcWidth = mask[0].length;
    final result = List.generate(height, (_) => List.filled(width, 0.0));
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final srcX = (x / width) * srcWidth;
        final srcY = (y / height) * srcHeight;
        final x0 = srcX.floor();
        final y0 = srcY.floor();
        final x1 = (x0 + 1).clamp(0, srcWidth - 1);
        final y1 = (y0 + 1).clamp(0, srcHeight - 1);
        final dx = srcX - x0;
        final dy = srcY - y0;
        result[y][x] =
            (1 - dx) * (1 - dy) * mask[y0][x0] +
            dx * (1 - dy) * mask[y0][x1] +
            (1 - dx) * dy * mask[y1][x0] +
            dx * dy * mask[y1][x1];
      }
    }
    return result;
  }

  List<List<double>> _blurMask(List<List<double>> mask, (int, int) blurSize) {
    final height = mask.length;
    final width = mask[0].length;
    final result = List.generate(height, (_) => List.filled(width, 0.0));
    final kernelX = blurSize.$1;
    final kernelY = blurSize.$2;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double sum = 0.0;
        int count = 0;
        for (int dy = -kernelY ~/ 2; dy <= kernelY ~/ 2; dy++) {
          for (int dx = -kernelX ~/ 2; dx <= kernelX ~/ 2; dx++) {
            final ny = (y + dy).clamp(0, height - 1);
            final nx = (x + dx).clamp(0, width - 1);
            sum += mask[ny][nx];
            count++;
          }
        }
        result[y][x] = sum / count;
      }
    }
    return result;
  }
}
