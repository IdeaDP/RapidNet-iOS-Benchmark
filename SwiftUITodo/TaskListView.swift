// //
// //  TaskListView.swift
// //  SwiftUITodo
// //
// //  Created by Suyeol Jeon on 03/06/2019.
// //  Copyright © 2019 Suyeol Jeon. All rights reserved.
// //

// import SwiftUI

// struct TaskListView: View {
//   @EnvironmentObject var userData: UserData
//   @State var draftTitle: String = ""
//   @State var isEditing: Bool = false

//   var body: some View {
//     List {
//       TextField($draftTitle, placeholder: Text("Create a New Task..."), onCommit: self.createTask)
//       ForEach(self.userData.tasks) { task in
//         TaskItemView(task: task, isEditing: self.$isEditing)
//       }
//     }
//     .navigationBarTitle(Text("Tasks 👀"))
//     .navigationBarItems(trailing: Button(action: { self.isEditing.toggle() }) {
//       if !self.isEditing {
//         Text("Edit")
//       } else {
//         Text("Done").bold()
//       }
//     })
//   }

//   private func createTask() {
//     let newTask = Task(title: self.draftTitle, isDone: false)
//     self.userData.tasks.insert(newTask, at: 0)
//     self.draftTitle = ""
//   }
// }



import SwiftUI
import CoreML

struct TaskListView: View {
    @State private var resultText = "Ready to benchmark RapidNet"
    
    var body: some View {
        VStack(spacing: 20) {
            Text(resultText)
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding()
            
            Button("Run Benchmark") { runBenchmark() }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
        }
    }
    
    func runBenchmark() {
        resultText = "Initializing..."
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all 
                
                // Load the auto-generated model class
                let rapidnet = try RapidNet_M(configuration: config)
                let coreMLModel = rapidnet.model 
                
                // Dynamically fetch the input tensor name (avoids naming mismatch errors)
                guard let inputName = coreMLModel.modelDescription.inputDescriptionsByName.keys.first else { return }
                
                let multiArray = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
                let featureProvider = try MLDictionaryFeatureProvider(dictionary: [inputName: multiArray])
                
                DispatchQueue.main.async { resultText = "Warming up NPU..." }
                for _ in 0..<50 { _ = try coreMLModel.prediction(from: featureProvider) }
                
                DispatchQueue.main.async { resultText = "Benchmarking..." }
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<100 { _ = try coreMLModel.prediction(from: featureProvider) }
                let totalTime = CFAbsoluteTimeGetCurrent() - start
                
                let avgMs = (totalTime / 100.0) * 1000.0
                DispatchQueue.main.async {
                    self.resultText = String(format: "Avg Inference: %.2f ms\nFPS: %.1f", avgMs, 1000/avgMs)
                }
            } catch {
                DispatchQueue.main.async { self.resultText = "Error: \(error.localizedDescription)" }
            }
        }
    }
}