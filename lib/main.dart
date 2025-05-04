import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'detection_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]).then((
    _,
  ) {
    runApp(const MyApp());
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '物件辨識應用程式',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const DetectionPage(),
    );
  }
}
