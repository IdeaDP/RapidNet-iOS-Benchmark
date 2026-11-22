"""
Deep Focus - convert the four ImageNet RapidNet models into Core ML *classifiers* with FLEXIBLE input size.

  python convert_classifiers_flexible.py            (run from RapidNet_project_base\\SwiftUITodo\\tools)

Reads   ..\\..\\RapidNet\\RapidNet_{Ti,S,M,B}.pth  and  ..\\..\\RapidNet-Official\\models\\rapidnet.py
Writes  ..\\SwiftUITodo\\RapidNet_{Ti,S,M,B}_Classifier.mlmodel

Each model:
  input  "image"      : RGB image, any of the sizes in SIZES (square), e.g. 128 ... 384 px
  output "classLabel" : best ImageNet class index as text ("0" ... "999")
  output "classProbs" : probability of all 1000 classes
ImageNet normalisation + softmax are inside the model; weights are float16 (half size, < 100 MB each).
The benchmark models (RapidNet_Ti.mlmodel ...) are NOT touched, so benchmark results stay comparable.
"""
import argparse, sys
from pathlib import Path
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent                                   # RapidNet_project_base
SIZES = [128, 160, 192, 224, 256, 288, 320, 384]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class Deployable(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
    def forward(self, x):
        return torch.softmax(self.net((x - self.mean) / self.std), dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["Ti", "S", "M", "B"])
    ap.add_argument("--weights-dir", default=str(BASE / "RapidNet"))
    ap.add_argument("--rapidnet-repo", default=str(BASE / "RapidNet-Official"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "SwiftUITodo"))
    ap.add_argument("--random-init", action="store_true", help="testing only: skip loading weights")
    args = ap.parse_args()

    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    sys.path.insert(0, args.rapidnet_repo)
    from models import rapidnet as R

    for v in args.variants:
        print(f"\n=== RapidNet-{v} ===")
        model = getattr(R, f"RapidNet_{v}")()
        if not args.random_init:
            ck = torch.load(Path(args.weights_dir) / f"RapidNet_{v}.pth", map_location="cpu", weights_only=False)
            model.load_state_dict(ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck)
        model.eval()
        fused = R.reparameterize_model(model).eval()      # also averages the distillation head into one output
        with torch.no_grad():
            for s in (SIZES[0], 224, SIZES[-1]):           # the network itself must accept every size
                x = torch.rand(1, 3, s, s)
                d = (model(x) - fused(x)).abs().max().item()
                print(f"  {s}x{s}: fused vs original max |diff| = {d:.2e}")
        traced = torch.jit.trace(Deployable(fused).eval(), torch.rand(1, 3, 224, 224))
        shapes = ct.EnumeratedShapes(shapes=[(1, 3, s, s) for s in SIZES], default=(1, 3, 224, 224))
        ml = ct.convert(
            traced,
            inputs=[ct.ImageType(name="image", shape=shapes, scale=1 / 255.0, bias=[0, 0, 0],
                                 color_layout=ct.colorlayout.RGB)],
            classifier_config=ct.ClassifierConfig(class_labels=[str(i) for i in range(1000)],
                                                  predicted_feature_name="classLabel"),
            convert_to="neuralnetwork")
        spec = ml.get_spec()
        if spec.description.predictedProbabilitiesName != "classProbs":
            ct.utils.rename_feature(spec, spec.description.predictedProbabilitiesName, "classProbs")
            ml = ct.models.MLModel(spec, skip_model_load=True)
        ml = quantization_utils.quantize_weights(ml, nbits=16)
        ml.short_description = f"RapidNet-{v} ImageNet-1K classifier, flexible input {SIZES[0]}-{SIZES[-1]} px"
        ml.author = "Deep Focus - EN3160"
        ml.user_defined_metadata["sizes"] = ",".join(map(str, SIZES))
        out = Path(args.out_dir) / f"RapidNet_{v}_Classifier.mlmodel"
        ml.save(str(out))
        print(f"  saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
