import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'dart:math';
// import 'ops.dart';
// import 'package:loredart_tensor/loredart_tensor.dart';

class OnnxSegmenter {
  OrtSession? _session;
  final int _modelInputSize = 640;
  final int _numMask = 32;
  final double _iouThreshold = 0.5;
  final double _confThreshold = 0.7;
  late int _imgWidth;
  late int _imgHeight;
  late final Float32List _inputBuffer = Float32List(
    1 * 3 * _modelInputSize * _modelInputSize,
  );
  Future<void> loadModel() async {
    try {
      OrtEnv.instance.init();
      const assetFileName = 'assets/models/yolo11nseg640nails.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(
        bytes,
        OrtSessionOptions()
          ..setIntraOpNumThreads(4)
          ..setInterOpNumThreads(2),
      );
      print('ONNXSEG模型載入成功');
    } catch (e) {
      print('ONNXSEG模型載入失敗: $e');
    }
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
      print('模型推理失敗: $e');
      return [];
    }
  }

  List<List<double>> _transpose(List<List<double>> matrix) {
    return List.generate(
      matrix[0].length,
      (i) => matrix.map((row) => row[i]).toList(),
    );
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

  List<List<double>> extractBoxes(List<List<double>> boxPredictions) {
    var boxes = boxPredictions.map((pred) => pred.sublist(0, 4)).toList();
    boxes = rescaleBoxes(
      boxes,
      (_modelInputSize, _modelInputSize),
      (_imgHeight, _imgWidth),
    );
    boxes = xywh2xyxy(boxes);
    boxes =
        boxes
            .map(
              (box) => [
                box[0].clamp(0, _imgWidth.toDouble()).toDouble(),
                box[1].clamp(0, _imgHeight.toDouble()).toDouble(),
                box[2].clamp(0, _imgWidth.toDouble()).toDouble(),
                box[3].clamp(0, _imgHeight.toDouble()).toDouble(),
              ],
            )
            .toList();
    return boxes;
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

  (List<List<double>>, List<double>, List<int>, List<List<double>>)
  _processBoxOutput(dynamic boxOutput) {
    final box = boxOutput.value as List;
    final predictions = _transpose(
      box[0],
    ); // Shape: [num_predictions, num_classes + 4 + num_masks]
    final numClasses = predictions[0].length - _numMask - 4;
    final scores =
        predictions
            .map((pred) => pred.sublist(4, 4 + numClasses).reduce(max))
            .toList();
    final filtered = <List<double>>[];
    final filteredScores = <double>[];
    for (int i = 0; i < predictions.length; i++) {
      if (scores[i] > _confThreshold) {
        filtered.add(predictions[i]);
        filteredScores.add(scores[i]);
      }
    }

    if (filtered.isEmpty) {
      return ([], [], [], []);
    }
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

    var boxes = extractBoxes(boxPredictions);
    final indices = nms(boxes, filteredScores, _iouThreshold);
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
    return maskOutput
        .map((mask) => mask.expand((row) => row).toList())
        .toList();
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

  List<List<List<double>>> processMaskOutput(
    List<List<double>> maskPredictions,
    List<List<double>> boxes,
    dynamic maskOutput,
  ) {
    if (maskPredictions.isEmpty) return [];

    List<List<List<double>>> processedMaskOutput = [];
    if (maskOutput is OrtValueTensor) {
      final maskData = maskOutput.value as List;
      // 檢查是否有批次維度
      processedMaskOutput = maskData[0];
    }
    final maskHeight = processedMaskOutput[0].length;
    final maskWidth = processedMaskOutput[0][0].length;
    // Compute masks: sigmoid(mask_predictions @ mask_output)
    final masks = _matrixMultiply(
      maskPredictions,
      _flattenMaskOutput(processedMaskOutput),
    );
    print(masks.length);
    print(masks[0].length);
    final reshapedMasks = _reshapeMasks(masks, maskHeight, maskWidth);

    // Downscale boxes
    final scaleBoxes = rescaleBoxes(
      boxes,
      (_imgHeight, _imgWidth),
      (maskHeight, maskWidth),
    );

    // Create mask maps
    final maskMaps = List.generate(
      scaleBoxes.length,
      (_) => List.generate(_imgHeight, (_) => List.filled(_imgWidth, 0.0)),
    );
    final blurSize = (_imgWidth ~/ maskWidth, _imgHeight ~/ maskHeight);

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

      // Threshold mask
      for (int y = 0; y < y2 - y1; y++) {
        for (int x = 0; x < x2 - x1; x++) {
          if (y + y1 < _imgHeight && x + x1 < _imgWidth) {
            maskMaps[i][y + y1][x + x1] = cropMask[y][x] > 0.5 ? 1.0 : 0.0;
          }
        }
      }
    }

    return maskMaps;
  }

  void printUniqueValues(List<List<double>> list) {
    // 展平二維列表並轉為 Set 以獲取唯一值
    Set<double> uniqueValues = list.expand((row) => row).toSet();

    // 列印所有唯一值
    print('Unique values: $uniqueValues');
  }

  Future<img.Image?> detect(img.Image image) async {
    final originalWidth = image.width.toDouble();
    final originalHeight = image.height.toDouble();
    img.Image outputImage = image;
    print('原始影像大小: $originalWidth x $originalHeight');
    _imgWidth = originalWidth.toInt();
    _imgHeight = originalHeight.toInt();
    final resizedImage = img.copyResize(
      image,
      width: _modelInputSize,
      height: _modelInputSize,
      interpolation: img.Interpolation.linear,
    );
    _fillInputBuffer(resizedImage);
    final outputs = await _runInference();
    final (boxes, scores, classid, maskpred) = _processBoxOutput(outputs[0]);
    // print(boxes.length);
    printUniqueValues(maskpred);
    final maskMaps = processMaskOutput(maskpred, boxes, outputs[1]);
    // printUniqueValues(maskMaps[0]);
    // print(maskMaps[0].length);
    for (int i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final mask = maskMaps[i];

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
      final mk = img.Image(
        width: outputImage.width,
        height: outputImage.height,
        format: img.Format.uint8,
        numChannels: 3,
      );
      for (int y = 0; y < mk.height; y++) {
        for (int x = 0; x < mk.width; x++) {
          final maskValue = (mask[y][x] * 255.0).clamp(0, 255).toInt();
          mk.setPixelRgb(x, y, maskValue, 0, 0);
        }
      }
      outputImage = addWeighted(outputImage, 1.0, mk, 1.0, 0.0);
    }
    return outputImage;
  }

  img.Image addWeighted(
    img.Image src1,
    double alpha,
    img.Image src2,
    double beta,
    double gamma,
  ) {
    final dst = img.Image(width: src1.width, height: src1.height);
    for (int y = 0; y < src1.height; y++) {
      for (int x = 0; x < src1.width; x++) {
        final p1 = src1.getPixel(x, y);
        final p2 = src2.getPixel(x, y);
        final r = (alpha * p1.r + beta * p2.r + gamma).clamp(0, 255).toInt();
        final g = (alpha * p1.g + beta * p2.g + gamma).clamp(0, 255).toInt();
        final b = (alpha * p1.b + beta * p2.b + gamma).clamp(0, 255).toInt();
        dst.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return dst;
  }

  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }
}
