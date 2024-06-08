from absl import app
import distance
import embedding
import util
import numpy as np


def run(encoder, class_type, formula, sample_size, effect, power, real_image_path, generated_image_path, batch_size = 1):
    return util.compute_metric(encoder, class_type, formula, sample_size, effect, power, 
                               real_image_path, generated_image_path, int(batch_size))
    
def main(argv):
    
    _, encoder, class_type, formula, sample_size, effect, power, real_image_path, generated_image_path, batch_size = argv    
    print("The metrics value is: "f" {util.compute_metric(encoder, class_type, formula, sample_size, effect, power, 
                               real_image_path, generated_image_path, int(batch_size))}")

if __name__ == "__main__":
    app.run(main)