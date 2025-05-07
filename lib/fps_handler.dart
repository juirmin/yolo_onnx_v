import 'dart:async';

class FPSHandler {
  int _frameCount = 0;
  Timer? _timer;

  void start(void Function(int) onFpsUpdate) {
    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      onFpsUpdate(_frameCount);
      _frameCount = 0;
    });
  }

  void stop() {
    _timer?.cancel();
  }

  void incrementFrameCount() {
    _frameCount++;
  }
}
