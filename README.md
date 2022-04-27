
## 1. Datasets

#### Speech ehancement datasets (sorted by usage frequency in paper)

### English 



|   **Name**   |                             **Source**                             | **Hours** |
| :----------: | :----------------------------------------------------------------: | :-------: |
| Dataset by University of Edinburgh | [https://datashare.ed.ac.uk](https://datashare.ed.ac.uk/handle/10283/2791) |   39h(141165.816338)   |
| VCTK(2009) | [https://datashare.ed.ac.uk](https://datashare.ed.ac.uk/handle/10283/3443) |   82h(297552.223038)   |
| LibriSpeech  | [http://www.openslr.org](http://www.openslr.org/12)              |   983h(3539995.721648)    |
| Common Voice | [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org) |   1837h(6615925.423006)   |
| The VoxCeleb1 Dataset | [https://www.robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |   x   |
| The VoxCeleb2 Dataset | [https://www.robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) |   x   |



### German

|   **Name**   |                             **Source**                              | **Hours** |
| :----------: | :-----------------------------------------------------------------: | :-------: |
| Common Voice | [https://commonvoice.mozilla.org/](https://commonvoice.mozilla.org) |   832h (2995609.871861)   |


#### Augmentation noise sources (sorted by usage frequency in paper)


|   **Name**   |                             **Source**                              | **Hours** |
| :----------: | :-----------------------------------------------------------------: | :-------: |
| DEMAND | [https://zenodo.org](https://zenodo.org/record/1227121#.YMK7OJMzZNx) |   8h(28800.384000)     |
| 100 Noise | [http://web.cse.ohio-state.edu](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip) |   293s(293.299375)   |
| RIRS_NOISES| [https://www.openslr.org](https://www.openslr.org/resources/28/rirs_noises.zip) |   27h(97661.178407)    |
| QUT-NOISE| [https://research.qut.edu.au](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols) |   27h (98262.946746)   |
| MUSAN | [https://www.openslr.org](https://www.openslr.org/17) |   48h (175827.483386)   |


#### Audio data augmentation

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [Data simulation](https://github.com/funcwj/setk/tree/master/doc/data_simu) | Python | Add reverberation, noise or mix speaker. |
| [audio-SNR](https://github.com/Sato-Kunihiko/audio-SNR) | Python | Mixing an audio file with a noise file at any Signal-to-Noise Ratio. |


## 2. 논문
1. Transformer
   * [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
   * [Attention Is All You Need - 나동빈](https://www.youtube.com/watch?v=AA621UofTUA)
   * [GAN: Generative Adversarial Networks (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AVvlDmhHgC4&t=1245s)

2. Speech Enhancement GAN
   * [Improving GANs for Speech Enhancement 논문](https://arxiv.org/pdf/2001.05532.pdf)
   * [Improving GANs for Speech Enhancement (페이퍼 관련 소스)](https://github.com/pquochuy/sasegan)
   * [Self Attention GAN for Speech Enhancement 논문](https://arxiv.org/pdf/2010.09132.pdf)
   * [Self Attention GAN for Speech Enhancement in Tensorflow 2 (페이퍼 관련 소스)](https://github.com/usimarit/sasegan)
   * [Spectral Normalization 논문](https://openreview.net/pdf?id=B1QRgziT-) 
   * [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification 논문](https://arxiv.org/pdf/1608.04363.pdf)
   * [To prune, or not to prune: exploring the efficacy of pruning for model compression 논문](https://arxiv.org/pdf/1710.01878.pdf)
   * [CMGAN: Conformer-based Metric GAN for Speech Enhancement](https://arxiv.org/pdf/2203.15149v1.pdf)

2. Data Augmentation
   * [SpecAugment](https://arxiv.org/pdf/1904.08779.pdf)
   * [SpecAugment: A New Data Augmentation Method for Automatic Speech Recognition - Google AI Blog] (https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html)
   
   
3. End to End
   * [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
   * [Convolution-augmented Transformer for Speech Recognition, 2020](https://arxiv.org/pdf/2005.08100.pdf)
   * [EFFICIENT CONFORMER, 2021](https://arxiv.org/pdf/2005.08100.pdf)
   * [Efficient Conformer (2021 페이퍼 관련 소스)](https://github.com/burchim/EfficientConformer)
   * [Openspeech seminar](https://github.com/sooftware/Speech-Recognition-Tutorial/tree/master/seminar)

3. [Awesome Speech Enhancement](https://github.com/nanahou/Awesome-Speech-Enhancement#Overview)


# 1.음성처리

## 1. 블러그
1. [Speech Processing for Machine Learning: Filter backs, Mel-Frequency Cepstral Coefficients(MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
2. [RNN transducer] (https://lorenlugosch.github.io/posts/2020/11/transducer/)

## 2. Tensorflow
1. [tf.data : TensorFlow 입력 파이프 라인 빌드](https://www.tensorflow.org/guide/data)
2. [tf.data API로 성능 향상하기](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/data_performance.ipynb?hl=ko)
3. 분산훈련
   * [분산 훈련 / strategy.num_replicas_in_sync](https://ahnjg.tistory.com/34)
   * [케라스를 사용한 분산 훈련 / MirroredStrategy ](https://www.tensorflow.org/tutorials/distribute/keras)
   * [텐서플로로 분산 훈련하기](https://colab.research.google.com/github/jiyongjung0/tf-docs/blob/distribute_strategy/site/ko/beta/guide/distribute_strategy.ipynb)
   * [학습 루프와 함께 tf.distribute.Strategy 사용하기](https://limjun92.github.io/assets/TensorFlow%202.0%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC/5.%20%EB%B6%84%EC%82%B0%ED%98%95%20%ED%95%99%EC%8A%B5/%5B%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC2%5D%20%ED%95%99%EC%8A%B5%20%EB%A3%A8%ED%94%84%EC%99%80%20%ED%95%A8%EA%BB%98%20tf.distribute.Strategy%20%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/)
4. [체크포인트 훈련하기](https://www.tensorflow.org/guide/checkpoint?hl=ko)
5. [tf.function으로 성능 향상하기](https://www.tensorflow.org/guide/function?hl=ko)

## 3. Audio recognition using Tensorflow Lite

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
14. Quantization
  * [Inside TensorFlow: TF Model Optimization Toolkit (Quantization and Pruning)](https://www.youtube.com/watch?v=4iq-d2AmfRU)
  * [TensorFlow Model Optimization Toolkit — Pruning API](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)
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

16. GitHub: 
  * [Deep Learning for Audio with TFLite support to run on Android](https://github.com/dhiraa/shabda)
  * [awesome-tensorflow-lite](https://github.com/margaretmz/awesome-tensorflow-lite)
  
## 4.Voice Filter 

1. [VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition](https://google.github.io/speaker-id/publications/VoiceFilter-Lite/)

# . Tensorflow
1. [tf.data : TensorFlow 입력 파이프 라인 빌드](https://www.tensorflow.org/guide/data)

# 2.Speech Recognition

## 1.toolkit
1. [End-to-End Speech Processing Toolkit](https://github.com/espnet/espnet)
2. [Openspeech](https://openspeech-team.github.io/openspeech/)
3.
