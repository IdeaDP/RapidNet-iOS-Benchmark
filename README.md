# RapidNet-M iPhone benchmark

The app runs the bundled RapidNet-M model with Core ML on the iPhone. It runs 50 warm-up predictions and times the next 100 on a synthetic Float32 input. The results show average prediction latency and throughput.

## Build with GitHub Actions

1. Copy the contents of this folder into the root of your GitHub clone and commit the changes. Include the hidden `.github` folder, `SwiftUITodo.xcodeproj/project.pbxproj`, and `SwiftUITodo/RapidNet_M.mlmodel`. If your clone has an older IPA build workflow under another filename in `.github/workflows`, replace that workflow with this copy's `build.yml`. Use Git or GitHub Desktop to upload the model, which exceeds GitHub's browser upload limit.
2. Open **Actions → Build iOS IPA**. Select **Run workflow** if a push hasn't already started the build.
3. Open the successful run, download the **BenchmarkApp-IPA** artifact, and extract `BenchmarkApp.ipa` from the downloaded ZIP.

## Install from Windows

1. Follow the [AltStore Classic Windows setup](https://faq.altstore.io/altstore-classic/how-to-install-altstore-windows) to install AltServer on Windows and AltStore on your iPhone. Enable Developer Mode when prompted.
2. Keep the iPhone connected to the PC with AltServer running. Transfer `BenchmarkApp.ipa` to the iPhone's Files app, open **AltStore → My Apps**, tap **+**, and select the IPA. AltStore signs the app with your Apple Account. With a free account, refresh the app within seven days to keep using it.
3. Launch the installed app and tap **Run Benchmark**.
