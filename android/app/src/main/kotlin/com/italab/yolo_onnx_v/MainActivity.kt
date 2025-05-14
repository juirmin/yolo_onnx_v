package com.italab.yolo_onnx_v

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugins.GeneratedPluginRegistrant
import android.util.Log
import com.king.libyuv.LibYuvUtils
import java.nio.ByteBuffer
import java.io.ByteArrayOutputStream

class MainActivity : FlutterActivity() {
    companion object {
        private const val CHANNEL = "com.italab.yolo_onnx_v/yuv_converter"
        private const val TAG = "YuvConverter"
    }

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        GeneratedPluginRegistrant.registerWith(flutterEngine)
        Log.d(TAG, "註冊 YUV 轉換 MethodChannel")

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "convertYUVtoRGB" -> {
                        try {
                            // 獲取從 Flutter 傳來的 YUV 數據
                            val planes = call.argument<List<ByteArray>>("planes")
                            val strides = call.argument<List<Int>>("strides")
                            val width = call.argument<Int>("width")
                            val height = call.argument<Int>("height")

                            if (planes == null || strides == null || width == null || height == null) {
                                result.error("INVALID_ARGS", "無效的參數", null)
                                return@setMethodCallHandler
                            }

                            Log.d(TAG, "處理圖像大小: ${width}x${height}")
                            Log.d(TAG, "平面數量: ${planes.size}")
                            for (i in planes.indices) {
                                Log.d(TAG, "平面 $i 大小: ${planes[i].size} 字節")
                            }
                            Log.d(TAG, "步長信息: $strides")

                            // 使用 LibYuv 處理 YUV 轉 RGB
                            val yRowStride = strides[0]
                            val uRowStride = strides[2]
                            val vRowStride = strides[4]
                            val uPixelStride = strides[3]
                            val vPixelStride = strides[5]

                            // 轉換 YUV 為 RGB
                            val rgbData = convertYuvToRgb(
                                planes[0], planes[1], planes[2],
                                yRowStride, uRowStride, vRowStride,
                                uPixelStride, vPixelStride,
                                width, height
                            )

                            // 返回結果
                            result.success(rgbData)
                        } catch (e: Exception) {
                            Log.e(TAG, "YUV 轉換錯誤: ${e.message}", e)
                            result.error("CONVERSION_ERROR", e.message, null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    /**
     * 使用 LibYuv 將 YUV 數據轉換為 RGB
     */
    private fun convertYuvToRgb(
        yPlane: ByteArray, uPlane: ByteArray, vPlane: ByteArray,
        yStride: Int, uStride: Int, vStride: Int,
        uvPixelStrideU: Int, uvPixelStrideV: Int,
        width: Int, height: Int
    ): ByteArray {
        try {
            // 檢查是否為 NV21 或 NV12 格式
            val isNV21 = uvPixelStrideU == 2 && uvPixelStrideV == 2
            val isNV12 = uvPixelStrideU == 1 && uvPixelStrideV == 1

            // 準備 I420 格式數據 (YUV 4:2:0 planar)
            val i420Data = when {
                isNV21 -> {
                    // 從 NV21 轉換為 I420
                    val i420 = ByteArray(width * height * 3 / 2)
                    LibYuvUtils.nv21ToI420(
                        yPlane, yStride,
                        uPlane, uStride,
                        i420, width, 
                        i420, width / 2, 
                        i420, width / 2,
                        width, height
                    )
                    i420
                }
                else -> {
                    // 將三個平面合併成 I420 格式
                    val i420 = ByteArray(width * height * 3 / 2)
                    System.arraycopy(yPlane, 0, i420, 0, width * height)
                    
                    // 對 U 和 V 平面進行重組 (考慮 pixel stride)
                    val uvWidth = width / 2
                    val uvHeight = height / 2
                    val uOffset = width * height
                    val vOffset = width * height + uvWidth * uvHeight
                    
                    // 複製 U 平面
                    if (uvPixelStrideU == 1) {
                        // U 平面連續存儲
                        System.arraycopy(uPlane, 0, i420, uOffset, uvWidth * uvHeight)
                    } else {
                        // U 平面有間隔 (pixel stride > 1)
                        for (row in 0 until uvHeight) {
                            for (col in 0 until uvWidth) {
                                val srcPos = row * uStride + col * uvPixelStrideU
                                val dstPos = uOffset + row * uvWidth + col
                                if (srcPos < uPlane.size) {
                                    i420[dstPos] = uPlane[srcPos]
                                }
                            }
                        }
                    }
                    
                    // 複製 V 平面
                    if (uvPixelStrideV == 1) {
                        // V 平面連續存儲
                        System.arraycopy(vPlane, 0, i420, vOffset, uvWidth * uvHeight)
                    } else {
                        // V 平面有間隔 (pixel stride > 1)
                        for (row in 0 until uvHeight) {
                            for (col in 0 until uvWidth) {
                                val srcPos = row * vStride + col * uvPixelStrideV
                                val dstPos = vOffset + row * uvWidth + col
                                if (srcPos < vPlane.size) {
                                    i420[dstPos] = vPlane[srcPos]
                                }
                            }
                        }
                    }
                    i420
                }
            }

            // 將 I420 轉換為 ARGB
            val rgbData = ByteArray(width * height * 4) // ARGB 格式, 每個像素 4 bytes
            LibYuvUtils.i420ToArgb(
                i420Data, width,
                i420Data, width / 2,
                i420Data, width / 2,
                rgbData, width * 4,
                width, height
            )

            // 將 ARGB 轉換為 RGB (移除 Alpha 通道)
            val rgb = ByteArray(width * height * 3)
            var rgbIndex = 0
            var argbIndex = 0
            for (i in 0 until width * height) {
                // 跳過 Alpha 通道
                argbIndex++
                // 複製 RGB 通道
                rgb[rgbIndex++] = rgbData[argbIndex++] // R
                rgb[rgbIndex++] = rgbData[argbIndex++] // G
                rgb[rgbIndex++] = rgbData[argbIndex++] // B
            }

            return rgb
        } catch (e: Exception) {
            Log.e(TAG, "轉換YUV到RGB失敗", e)
            throw e
        }
    }
}