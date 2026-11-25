# RapidNet iPhone app

The SwiftUI app runs RapidNet Core ML models on an iPhone. It includes a model benchmark and two live-camera modes: semantic segmentation and object detection.

## App options

### Choose a model

The model picker offers each bundled RapidNet variant: Ti, S, M, and B. The selected variant is used by **Run Benchmark**. The live camera modes currently use RapidNet-M.

### Run Benchmark

The benchmark reports average prediction latency in milliseconds and throughput in predictions per second. It also shows the compute setting, number of retained timing samples, and the iPhone's thermal state at the start and end.

The benchmark uses a synthetic Float32 input, not camera frames or an ImageNet validation set. It performs 50 rounds of 50 predictions: one warm-up prediction per round is excluded, and the 10 slowest and 10 fastest measured predictions in each round are trimmed. This is a device speed benchmark, not a measure of model accuracy or a complete reproduction of a research-paper evaluation.

### Open Segmentation Camera

Uses the rear camera and RapidNet-M to produce a semantic class map for each frame. The app overlays the class colors on a square 512×512 camera crop and shows model time and processed FPS. Semantic segmentation assigns a class to image regions; it does not draw object bounding boxes.

### Open Detection Camera

Intended to run the RapidNet-M COCO object detector on the live rear-camera image. It draws a box, class name, and confidence score for each detected object, then shows detector time and processed FPS. This camera mode uses a separate detector model path from semantic segmentation.

**Current status:** The installed build has been observed showing segmentation from both camera buttons. Object detection is included in the app code, but verify that the Detection Camera shows labeled boxes before recording detection results.

### Camera access and controls

The first camera launch may ask for permission. Allow camera access in iPhone Settings if prompted. Tap **Done** to close either camera screen.

## Build with GitHub Actions

1. Copy the contents of this folder into the root of your GitHub clone and commit the changes. Include the hidden `.github` folder, `SwiftUITodo.xcodeproj/project.pbxproj`, and the required model and label assets under `SwiftUITodo`. If your clone has an older IPA build workflow under another filename in `.github/workflows`, replace that workflow with this copy's `build.yml`. Use Git or GitHub Desktop to upload large model files.
2. Open **Actions → Build iOS IPA**. Select **Run workflow** if a push hasn't already started the build.
3. Open the successful run, download the **BenchmarkApp-IPA** artifact, and extract `BenchmarkApp.ipa` from the downloaded ZIP.

## Install from Windows

1. Follow the [AltStore Classic Windows setup](https://faq.altstore.io/altstore-classic/how-to-install-altstore-windows) to install AltServer on Windows and AltStore on your iPhone. Enable Developer Mode when prompted.
2. Keep the iPhone connected to the PC with AltServer running. Transfer `BenchmarkApp.ipa` to the iPhone's Files app, open **AltStore → My Apps**, tap **+**, and select the IPA. AltStore signs the app with your Apple Account. With a free account, refresh the app within seven days to keep using it.
3. Launch the installed app. Choose a model and tap **Run Benchmark**, or open one of the camera modes.
