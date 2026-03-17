import coremltools as ct
import tensorflow as tf
import os

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Loaded {len(class_names)} classes: {class_names}")

# Load your Keras 3 model
model = tf.keras.models.load_model("roiswift_landmark_model_left.keras")
print("✅ Model loaded successfully")

# Save as SavedModel format
saved_model_dir = "temp_saved_model"
tf.saved_model.save(model, saved_model_dir)
print(f"✅ Saved to {saved_model_dir}")

# Convert to CoreML with correct input name
mlmodel = ct.convert(
    saved_model_dir,
    source="tensorflow",
    inputs=[ct.ImageType(
        name="inputs",  # This is the correct placeholder name
        shape=(1, 200, 200, 1),
        color_layout=ct.colorlayout.GRAYSCALE,
        scale=1/255.0,
        bias=[0]
    )],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    classifier_config=ct.ClassifierConfig(class_names)
)

# Set metadata
mlmodel.author = "ASL Translation System"
mlmodel.short_description = "ASL hand gesture classifier (200x200 grayscale)"
mlmodel.version = "1.0"

# Add descriptions
mlmodel.input_description["inputs"] = "Grayscale hand landmark image (200x200 pixels)"

# Describe outputs
for output in mlmodel.get_spec().description.output:
    if "classLabel" in output.name or "class" in output.name.lower():
        mlmodel.output_description[output.name] = "Predicted ASL letter"
    else:
        mlmodel.output_description[output.name] = "Probability distribution over ASL letters"

# Save
mlmodel.save("ASLClassifier_left.mlpackage")

print(f"\n✅ CoreML model saved to: ASLClassifier.mlpackage")
print(f"\nModel specifications:")
print(f"  - Input: 200x200 grayscale image (hand landmarks)")
print(f"  - Preprocessing: Normalized by 1/255 (matches Python script)")
print(f"  - Output: {len(class_names)} ASL classes")
print(f"  - Classes: {', '.join(class_names)}")

# Cleanup
import shutil
shutil.rmtree(saved_model_dir)
print("🧹 Cleaned up temporary files")
