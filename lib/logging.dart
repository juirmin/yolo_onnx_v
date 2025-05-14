import 'dart:async';
import 'dart:isolate';

class Logger {
  SendPort? _logSendPort;
  List<String> _logBuffer = [];

  void setup() {
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

  void log(String message) {
    if (_logSendPort != null) {
      _logSendPort!.send(message);
    } else {
      _logBuffer.add(message);
    }
  }
}
