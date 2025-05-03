import 'package:flutter/material.dart';
import 'detection_page.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: '物件識別', home: DetectionPage());
  }
}
