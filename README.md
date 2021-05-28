# 1.음성처리

## 1. 블러그
1. [Speech Processing for Machine Learning: Filter backs, Mel-Frequency Cepstral Coefficients(MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
 

## 2. Audio recognition using Tensorflow Lite

1. [tflite-support](https://github.com/tensorflow/tflite-support.git)

2. [오디오 분류기 통합](https://www.tensorflow.org/lite/tutorials/model_maker_audio_classification)
  -> [오디오 분류기 통합2](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)

3. [오디오 데이터 준비 및 증강](https://www.tensorflow.org/io/tutorials/audio?hl=ko)

4. [모델변환](https://www.tensorflow.org/lite/convert)

5. [모델최적화](https://www.tensorflow.org/lite/performance/model_optimization)

6. [TensorFlow Lite Model Maker를 사용하여 오디오 도메인에 대한 전이 학습](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)

7. [Tensorflow and Tensorflow Lite code in the context of audio processing (MFCC, RNN)](https://gist.github.com/padoremu/8288b47ce76e9530eb288d4eec2e0b4d)

8. [Audio recognition using Tensorflow Lite in Flutter applications](https://carolinamalbuquerque.medium.com/audio-recognition-using-tensorflow-lite-in-flutter-application-8a4ad39964ae)

9. Udacity - [TensorFlow Lite  e-Learning과정](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)

10. Coursera - [Device-based Models with TensorFlow Lite](https://www.coursera.org/learn/device-based-models-tensorflow#syllabus)

11. [A Definitive Guide for Audio Processing in Android with TensorFlow Lite Models](https://heartbeat.fritz.ai/a-definitive-guide-for-audio-processing-in-android-with-tensorflow-lite-models-d90de896f0c4)

12. [Android에서 커스텀 TensorFlow Lite 모델 사용](https://firebase.google.com/docs/ml/android/use-custom-models?hl=ko)

13. Tensorflow lite
  * [파이킴 runForMultipleInputsOutputs](https://pythonkim.tistory.com/134?category=703510)
  * [Tensorflow Lite android 기본 개인정리](https://wiserloner.tistory.com/1379)
  * [TensorFlow Tutorial](https://data-flair.training/blogs/tensorflow-tutorial/)

~~~

Bitmap bitmap = Bitmap.createScaledBitmap(yourInputImage, 224, 224, true);
ByteBuffer input = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder());
for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
        int px = bitmap.getPixel(x, y);

        // Get channel values from the pixel value.
        int r = Color.red(px);
        int g = Color.green(px);
        int b = Color.blue(px);

        // Normalize channel values to [-1.0, 1.0]. This requirement depends
        // on the model. For example, some models might require values to be
        // normalized to the range [0.0, 1.0] instead.
        float rf = (r - 127) / 255.0f;
        float gf = (g - 127) / 255.0f;
        float bf = (b - 127) / 255.0f;

        input.putFloat(rf);
        input.putFloat(gf);
        input.putFloat(bf);
    }
}
int bufferSize = 1000 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
interpreter.run(input, modelOutput);
...

~~~
14. [Tensorflow-lite signiture 분석](https://quizee-ab.tistory.com/14)

15. [Recognize Flowers with TensorFlow Lite on Android](https://developer.android.com/codelabs/recognize-flowers-with-tensorflow-on-android?hl=pt#0)

16. GitHub: [Deep Learning for Audio with TFLite support to run on Android](https://github.com/dhiraa/shabda)
