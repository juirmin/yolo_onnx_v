import 'package:flutter/material.dart';
import 'models.dart';
import 'dart:math';

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
    if (detections.isEmpty) return;
    final double scaleX = size.width / previewSize.width;
    final double scaleY = size.height / previewSize.height;
    final double scale = min(scaleX, scaleY);
    final double offsetX = (size.width - previewSize.width * scale) / 2;
    final double offsetY = (size.height - previewSize.height * scale) / 2;

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
          max(0.0, screenTop - textPainter.height - 4),
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
