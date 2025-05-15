import 'package:loredart_tensor/loredart_tensor.dart';
import 'package:onnxruntime/onnxruntime.dart';

dynamic squeeze(dynamic array) {
  if (array is OrtValue) {
    final listValue = array.value as List;
    final Tensor tensor = Tensor.constant(listValue);
    final squeezeData = squeeze(tensor);
    final squeezeList = squeezeData.value as List;
    return squeezeList;
  } else {
    throw Exception('Unsupported type: ${array.runtimeType}');
  }
}
