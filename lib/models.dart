class Detection {
  /// 邊界框 [left, top, right, bottom]
  final List<double> boundingBox;

  /// 信心分數
  final double confidence;

  /// 類別ID
  final int classId;

  /// 類別名稱
  final String className;

  Detection({
    required this.boundingBox,
    required this.confidence,
    required this.classId,
    this.className = '',
  });
}
